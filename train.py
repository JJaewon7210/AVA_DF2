# Standard library imports
import argparse
import logging
import platform
import math
import os
import random
import time
from copy import deepcopy
from threading import Thread
from pathlib import Path

# Related third party imports
import yaml
import torch
import matplotlib.pyplot as plt
import numpy as np
from torch.cuda import amp
import torch.optim as optim
from tqdm import tqdm

# Local application/library specific imports
from model.model import MTA_F3D_MODEL as Model
from utils.general import labels_to_class_weights, increment_path, init_seeds, \
    fitness, strip_optimizer, get_latest_run, check_dataset, check_file, check_git_status, check_img_size, \
    check_requirements, print_mutation, set_logging, colorstr, ConfigObject, non_max_suppression
from utils.wandb_logging.wandb_utils import WandbLogger, check_wandb_resume
from utils.scheduler import CosineAnnealingWarmupRestarts
from utils.torch_utils import ModelEMA, select_device, intersect_dicts, torch_distributed_zero_first, is_parallel
from utils.plots import plot_images, plot_labels, plot_results, plot_evolution

from utils.loss_ava import ComputeLossOTA,  WeightedMultiTaskLoss
from datasets.ava_dataset import AvaWithPseudoLabel, Ava
from datasets.yolo_datasets import DeepFasion2WithPseudoLabel, LoadImagesAndLabels, InfiniteDataLoader
from datasets.combined_dataset import CombinedDataset
from test_ava import test_ava
from test_df2 import test_df2

logger = logging.getLogger(__name__)

def main(hyp, opt, device, tb_writer):
    '''
    YOLOv7 Style Trainer
    '''
    save_dir, epochs, batch_size, weights = \
        Path(opt.save_dir), opt.epochs, opt.batch_size, opt.weights


    # 1. Directories
    wdir = save_dir / 'weights'
    wdir.mkdir(parents=True, exist_ok=True)  # make dir
    last = wdir / 'last.pt'
    best = wdir / 'best.pt'
    results_file = save_dir / 'results.txt'


    # 2. Save run settings
    with open(save_dir / 'hyp.yaml', 'w') as f:
        yaml.dump(hyp, f, sort_keys=False)
    with open(save_dir / 'opt.yaml', 'w') as f:
        yaml.dump(vars(opt), f, sort_keys=False)


    # 3. Configure
    plots = True
    cuda = device.type != 'cpu'
    init_seeds(1)
    opt_dict = opt.to_dict()  # data dict


    # 4. Logging- Doing this before checking the dataset. Might update opt_dict
    opt.hyp = hyp  # add hyperparameters
    run_id = torch.load(weights, map_location=device).get('wandb_id') if weights.endswith('.pt') and os.path.isfile(weights) else None
    wandb_logger = WandbLogger(opt, Path(opt.save_dir).stem, run_id, opt_dict)
    opt_dict = wandb_logger.data_dict
    if wandb_logger.wandb:
        weights, epochs, hyp = opt.weights, opt.epochs, opt.hyp  # WandbLogger might update weights, epochs if resuming


    # 5. Model 
    pretrained = weights.endswith('.pt')
    if pretrained:
        ckpt = torch.load(weights, map_location=device)  # load checkpoint
        model = Model(cfg=opt).to(device)
        exclude = ['anchor'] if (hyp.get('anchors')) and not opt.resume else []  # exclude keys
        state_dict = ckpt['model'].float().state_dict()  # to FP32
        state_dict = intersect_dicts(state_dict, model.state_dict(), exclude=exclude)  # intersect
        model.load_state_dict(state_dict, strict=False)  # load
        logger.info('Transferred %g/%g items from %s' % (len(state_dict), len(model.state_dict()), weights))  # report
    else:
        model = Model(cfg=opt).to(device)


    # 6. Dataset, Dataloader
    imgsz, imgsz_test = [check_img_size(x, 32) for x in opt.img_size]  # verify imgsz are gs-multiples
    dataset_df2 = DeepFasion2WithPseudoLabel(path=opt.train, img_size=imgsz, batch_size=opt.batch_size, 
                                             augment=False, hyp=hyp, rect=False, image_weights=opt.image_weights,
                                             cache_images=opt.cache_images, single_cls=opt.single_cls, 
                                             stride=32, pad=0.0, prefix='train: ')
    dataset_ava = AvaWithPseudoLabel (cfg=opt, split='train', only_detection=False)
    dataset = CombinedDataset(dataset_df2, dataset_ava)
    loader = torch.utils.data.DataLoader if opt.image_weights else InfiniteDataLoader
    dataloader = loader(dataset, batch_size=opt.batch_size, num_workers=opt.workers, collate_fn=CombinedDataset.collate_fn)
    num_batch = len(dataloader)  # number of batches
    
    if not opt.notest:
        logger.info('\n====> (For test) Loading LoadImagesAndLabels Dataset')
        testset_df2 = LoadImagesAndLabels(path=opt.val, img_size=imgsz_test, batch_size=opt.batch_size_test, 
                                                augment=False, hyp=opt.hyp, rect=False, image_weights=opt.image_weights,
                                                cache_images=opt.cache_images, single_cls=opt.single_cls, 
                                                stride=32, pad=0.0, prefix='val: ')
        logger.info('\n====> (For test) Loading Ava Dataset')
        testset_ava = Ava(cfg=opt, split='val', only_detection=False)
        loader = torch.utils.data.DataLoader if opt.image_weights else InfiniteDataLoader
        testloader_df2 = loader(testset_df2, batch_size=opt.batch_size_test, num_workers=opt.workers, collate_fn=LoadImagesAndLabels.collate_fn)
        testloader_ava = loader(testset_ava, batch_size=opt.batch_size_test, num_workers=opt.workers)
    else:
        logger.info('\n====> No test section during training model.')
        
    # 7. Optimizer, LR scheduler 
    optimizer = optim.Adam(model.parameters(), lr=hyp['lr0'], weight_decay=hyp['weight_decay'] )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=hyp['lrmax'], epochs = epochs, steps_per_epoch=num_batch)


    # 8. Resume
    start_epoch, best_fitness = 0, 1e+9
    if pretrained:
        # Optimizer
        if ckpt['optimizer'] is not None:
            optimizer.load_state_dict(ckpt['optimizer'])
            best_fitness = ckpt['best_fitness']
        # Results
        if ckpt.get('training_results') is not None:
            results_file.write_text(ckpt['training_results'])  # write results.txt
        # Epochs
        start_epoch = ckpt['epoch'] + 1
        if opt.resume:
            assert start_epoch > 0, '%s training to %g epochs is finished, nothing to resume.' % (weights, epochs)
        if epochs < start_epoch:
            logger.info('%s has been trained for %g epochs. Fine-tuning for %g additional epochs.' %
                        (weights, ckpt['epoch'], epochs))
            epochs += ckpt['epoch']  # finetune additional epochs

        del ckpt, state_dict


    # Start training ----------------------------------------------------------------------------------------------------
    scheduler.last_epoch = start_epoch - 1  # do not move
    torch.save(model, wdir / 'init.pt')
    scaler = amp.GradScaler(enabled=False)
    logger.info(f'Using {dataloader.num_workers} dataloader workers\n'
                f'Logging results to {save_dir}\n'
                f'Starting training for {epochs} epochs...')
    AVA_L = ComputeLossOTA(cfg=opt)
    LOSS = WeightedMultiTaskLoss(num_tasks=4)
    
    # Start epoch ------------------------------------------------------------------------------------------------------
    for epoch in range(start_epoch, epochs):  
        model.train()
        optimizer.zero_grad()
        pbar = enumerate(dataloader)
        pbar = tqdm(pbar, total=num_batch)  # progress bar
        logger.info(('\n' + '%10s' * 7) % ('gpu_mem', 'box', 'obj', 'act', 'clo', 'total', 'lr'))
        mloss = torch.zeros(4, device=device)  # mean losses
        mtotal_loss = torch.zeros(1, device=device) # mean total_loss (sum of losses)
        
        # Start batch ----------------------------------------------------------------------------------------------------
        for i, (item1, item2) in pbar:
            
            # Batch-01. Input data
            imgs, labels, paths, _shapes, features = item1
            imgs = imgs.to(device, non_blocking=True).float() / 255.0  # uint8 to float32, 0-255 to 0.0-1.0
            '''
             Explanation of variables in item1_batch:
               - imgs (torch.uint8): Image data with shape [B, 3, H, W] which has the scale of 0~255
               - labels (torch.float32): Label data with shape [num, 6], where num is the number of objects in the image.
                                        Each row in the labels contains: [Batch_num, class_num, x, y, h, w].
               - paths (tuple[str]): List of image paths with length B.
               - _shapes (tuple[tuple]): List of tuples, each containing (h0, w0), ((h / h0, w / w0), pad).
               - features (torch.float32): Features data with shape [B, 15, 7, 7], representing 3 (num_anchor) x 5 (bbox xywh + confidence score).
            '''
            
            clips, cls, boxes, feature_s, feature_m, feature_l = item2
            clips = clips.to(device, non_blocking=True)
            '''
             Explanation of variables in item2_batch:
               - clips (torch.float32): Video clips data with shape [B, 3, T, H, W] which has the scale of 0~1
               - cls (np.array, dtype('float32')): Array of class data with shape [B, 50, 80].
                                                  It contains up to 50 labels (from the beginning to the end).
               - boxes (np.array, dtype('float32')): Array of bounding box data with shape [B, 50, 4].
               - feature_s (np.array, dtype('float16')): Array of small feature data with shape [B, 3, 7, 7, 13].
               - feature_m (np.array, dtype('float16')): Array of medium feature data with shape [B, 3, 14, 14, 13].
               - feature_l (np.array, dtype('float16')): Array of large feature data with shape [B, 3, 28, 28, 13].
            '''
            
            # Concatenate 'imgs_duplicated' and 'clips' along the first dimension, which has the shape of [2B, 3, T, H, W]
            imgs_duplicated = imgs.unsqueeze(2).repeat((1, 1, clips.shape[2], 1, 1))
            model_input = torch.cat([imgs_duplicated, clips], dim=0)
            
            # Batch-02. Forward
            with amp.autocast(enabled=False):
                out_bboxs, out_clos, out_acts = model(model_input)
                
                out_bbox_infer, out_bbox_features = out_bboxs[0], out_bboxs[1]
                out_clo_infer, out_clo_features = out_clos[0], out_clos[1]
                out_act_infer, out_act_features = out_acts[0], out_acts[1]

                #TODO: Define the loss function
                out_bbox_AVA = [i[-batch_size:] for i in out_bbox_features]
                out_bbox_DF2 = [i[:batch_size] for i in out_bbox_features]
                out_act_AVA = [i[-batch_size:] for i in out_act_features]
                out_clo_DF2 = [i[:batch_size] for i in out_clo_features]
                
                _sum, losses = AVA_L.forward_ava(p_cls=out_act_AVA, p_bbox=out_bbox_AVA, t_cls=cls, t_bbox=boxes)
                lbox, lobj, lact, loss = torch.split(losses, 1)
                _sum, losses = AVA_L.forward_df2(p_cls=out_clo_DF2, p_bbox=out_bbox_DF2, targets=labels)
                _lbox, _lobj, lclo, loss = torch.split(losses, 1)
                total_loss = LOSS([lbox, lobj, lact, lclo])
                
                
            # Batch-03. Backward
            scaler.scale(total_loss).backward()
            scaler.step(optimizer)  # optimizer.step
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()

            # Batch-04. Print
            loss_item = torch.tensor([lbox, lobj, lact, lclo],device=device)
            
            if torch.all(torch.isfinite(loss_item)):
                mloss = (mloss * i + loss_item) / (i + 1)  # update mean losses
                mtotal_loss = (mtotal_loss * i + total_loss) / (i+1) # update mean total_loss
                
            gpu_memory_usage_gb = '%.3gG' % (torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0)
            lr = [x['lr'] for x in optimizer.param_groups]
            output_string = '%g/%g' % (epoch, epochs - 1)  # Display current epoch / total epochs
            output_string += ' ' + gpu_memory_usage_gb
            for loss_idx in range(len(mloss)):
                output_string += ' ' + '%10.4g' % mloss[loss_idx]
            output_string += ' ' + '%10.4g' % mtotal_loss
            output_string += ' ' + '%10.6g' % lr[0]
            pbar.set_description(output_string)
            
            cur_step = epoch * num_batch + i
            if (cur_step % opt.log_step) == 0:
                tags = ['train/box_loss', 'train/obj_loss', 'train/act_loss', 'train/clo_loss', 'lr']
                for x, tag in zip(list(mloss[:-1]) + lr, tags):
                    wandb_logger.log({tag: x})
                    
            # Batch-05. Plot
            if plots and i < 10:
                f_clo = save_dir / f'train_batch_clo{i}.jpg'  # filename
                f_act = save_dir / f'train_batch_act{i}.jpg'  # filename
                
                preds_clo = torch.cat((out_bbox_infer, out_clo_infer), dim=2)
                preds_clo = non_max_suppression(preds_clo, conf_thres=0.75, iou_thres=0.75)
                preds_clo = torch.cat(preds_clo, dim=0)
                preds_act = torch.cat((out_bbox_infer, out_act_infer), dim=2)
                preds_act = non_max_suppression(preds_act, conf_thres=0.75, iou_thres=0.75)
                preds_act = torch.cat(preds_act, dim=0)
                
                Thread(target=plot_images, args=(model_input[:, :, -1, :, :], preds_clo, None, f_clo), daemon=True).start()
                Thread(target=plot_images, args=(model_input[:, :, -1, :, :], preds_act, None, f_act), daemon=True).start()
                
            elif plots and i == 10 and wandb_logger.wandb:
                wandb_logger.log({"Mosaics": [wandb_logger.wandb.Image(str(x), caption=x.name) for x in
                                                save_dir.glob('train*.jpg') if x.exists()]})

            # End batch ----------------------------------------------------------------------------------------------------

        # End epoch --------------------------------------------------------------------------------------------------------
    
        # Start write ------------------------------------------------------------------------------------------------------
        final_epoch = epoch + 1 == epochs
        wandb_logger.current_epoch = epoch + 1
        
        if not opt.notest or final_epoch:
            results_ava, maps_ava, times_ava = test_ava(opt,
                                        batch_size=opt.batch_size_test,
                                        imgsz=imgsz_test,
                                        model=model,
                                        single_cls=opt.single_cls,
                                        dataloader=testloader_ava,
                                        save_dir=save_dir,
                                        verbose=final_epoch,
                                        plots=plots and final_epoch,
                                        wandb_logger=wandb_logger,
                                        compute_loss=False,
                                        is_coco=False,
                                        v5_metric=opt.v5_metric)
            
            results_df2, maps_df2, times_df2 = test_df2(opt,
                                        batch_size=opt.batch_size_test,
                                        imgsz=imgsz_test,
                                        model=model,
                                        single_cls=opt.single_cls,
                                        dataloader=testloader_df2,
                                        save_dir=save_dir,
                                        verbose=final_epoch,
                                        plots=plots and final_epoch,
                                        wandb_logger=wandb_logger,
                                        compute_loss=False,
                                        is_coco=False,
                                        v5_metric=opt.v5_metric)
        # Log
        tags = [
                'metrics/precision(AVA)', 'metrics/recall(AVA)', 'metrics/mAP_0.5(AVA)', 'metrics/mAP_0.5:0.95(AVA)',
                'val/box_loss(AVA)', 'val/obj_loss(AVA)', 'val/act_loss(AVA)',  # val loss
                'metrics/precision(DF2)', 'metrics/recall(DF2)', 'metrics/mAP_0.5(DF2)', 'metrics/mAP_0.5:0.95(DF2)',
                'val/box_loss(DF2)', 'val/obj_loss(DF2)', 'val/clo_loss(DF2)',  # val loss
                ]  # params
        
        # Write
        simple_tags = ['gpu_mem', 'box', 'obj', 'act', 'clo', 'total', 'lr']+[tag.split('/')[-1] for tag in tags]
        formatted_tags = [f"{tag:<{width}}" for tag, width in zip(simple_tags, [10]* (7+len(simple_tags)))] 
        header_line = " ".join(formatted_tags) + '\n'

        if not os.path.exists(results_file) or os.path.getsize(results_file) == 0: # Check if file is empty or does not exist
            with open(results_file, 'w') as f:
                f.write(header_line)

        with open(results_file, 'a') as f:
            f.write(output_string + '%10.4g' * 7 % results_ava + '%10.4g' * 7 % results_df2 +'\n')  # append metrics, val_loss
        
        for x, tag in zip(list(results_ava) + list(results_df2), tags):
            if tb_writer:
                tb_writer.add_scalar(tag, x, epoch)  # tensorboard
            if wandb_logger.wandb:
                wandb_logger.log({tag: x})  # W&B
                
        # Update best score (Define best_fitness as minimum loss)
        fi = mtotal_loss
        if fi < best_fitness:
            best_fitness = fi
        wandb_logger.end_epoch(best_result=best_fitness == fi)
        
        # Save model
        if (not opt.nosave) or (final_epoch and not opt.evolve):  # if save
            ckpt = {'epoch': epoch,
                    'best_fitness': best_fitness,
                    'training_results': results_file.read_text(),
                    'model': deepcopy(model.module if is_parallel(model) else model).half(),
                    'optimizer': optimizer.state_dict(),
                    'wandb_id': wandb_logger.wandb_run.id if wandb_logger.wandb else None}

            # Save last, best and delete
            torch.save(ckpt, last)
            if best_fitness == fi:
                torch.save(ckpt, best)
            if (best_fitness == fi) and (epoch >= 200):
                torch.save(ckpt, wdir / 'best_{:03d}.pt'.format(epoch))
            if epoch == 0:
                torch.save(ckpt, wdir / 'epoch_{:03d}.pt'.format(epoch))
            elif ((epoch+1) % 25) == 0:
                torch.save(ckpt, wdir / 'epoch_{:03d}.pt'.format(epoch))
            elif epoch >= (epochs-5):
                torch.save(ckpt, wdir / 'epoch_{:03d}.pt'.format(epoch))
            if wandb_logger.wandb:
                if ((epoch + 1) % opt.save_period == 0 and not final_epoch) and opt.save_period != -1:
                    wandb_logger.log_model(
                        last.parent, opt, epoch, fi, best_model=best_fitness == fi)
            del ckpt

        # End write --------------------------------------------------------------------------------------------------------
    # End training -----------------------------------------------------------------------------------------------------

if __name__ == '__main__':

    # Import configuration files
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    with open('cfg/deepfashion2.yaml', 'r') as f:
        _dict_df2 = yaml.safe_load(f)
        opt_df2 = ConfigObject(_dict_df2)
    
    with open('cfg/ava.yaml', 'r') as f:
        _dict_ava = yaml.safe_load(f)
        opt_ava = ConfigObject(_dict_ava)
        
    with open('cfg/model.yaml', 'r') as f:
        _dict_model = yaml.safe_load(f)
        opt_model = ConfigObject(_dict_model)
    
    with open('cfg/hyp.yaml', 'r') as f:
        hyp = yaml.safe_load(f)
    
    logger.info(colorstr('Hyperparameters: ') + ', '.join(f'{k}={v}' for k, v in hyp.items()))
    opt = ConfigObject({})
    opt.merge(opt_df2)
    opt.merge(opt_ava)
    opt.merge(opt_model)
    
    # Resume
    wandb_run = check_wandb_resume(opt)
    if opt.resume and not wandb_run:  # resume an interrupted run
        ckpt = opt.resume if isinstance(opt.resume, str) else get_latest_run()  # specified or most recent path
        assert os.path.isfile(ckpt), 'ERROR: --resume checkpoint does not exist'
        apriori = opt.global_rank, opt.local_rank
        with open(Path(ckpt).parent.parent / 'opt.yaml') as f:
            opt = argparse.Namespace(**yaml.load(f, Loader=yaml.SafeLoader))  # replace
        opt.weights, opt.resume, opt.batch_size, opt.global_rank, opt.local_rank = ckpt, True, opt.total_batch_size, *apriori  # reinstate
        logger.info('Resuming training from %s' % ckpt)
    else:
        # opt.hyp = opt.hyp or ('hyp.finetune.yaml' if opt.weights else 'hyp.scratch.yaml')
        opt.img_size.extend([opt.img_size[-1]] * (2 - len(opt.img_size)))  # extend to 2 sizes (train, test)
        opt.save_dir = increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok)  # increment run
    
    # Empty the cash for preventing 'cuda out of memory'
    torch.cuda.empty_cache()
    torch.cuda.reset_max_memory_allocated()
    torch.cuda.synchronize()
    
    main(hyp, opt, device = torch.device('cuda:0'), tb_writer = None)