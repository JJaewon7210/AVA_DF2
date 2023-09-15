# Standard library imports
import argparse
import logging
import os
import random
from copy import deepcopy
from threading import Thread
from pathlib import Path
import cv2

# Related third party imports
import yaml
import torch
import numpy as np
from torch.cuda import amp
from tqdm import tqdm
from timm.optim import create_optimizer_v2

# Local application/library specific imports

from model.YOWO import YOWO_CUSTOM as Model
from utils.general import labels_to_class_weights, increment_path, labels_to_image_weights, init_seeds, \
    get_latest_run, check_img_size, colorstr, ConfigObject, non_max_suppression
from utils.wandb_logging.wandb_utils import WandbLogger, check_wandb_resume
from utils.scheduler import CosineAnnealingWarmupRestarts
from utils.torch_utils import ModelEMA, intersect_dicts, is_parallel
from utils.plots import plot_images, read_labelmap, un_normalized_images, plot_batch_image_from_preds, output_to_target

from utils.loss_ava import ComputeLoss,  WeightedMultiTaskLoss
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
    na = len(opt.MODEL.ANCHORS[0]) // 2 # number of anchors
    init_seeds(1)
    opt_dict = opt.to_dict()  # data dict
    labelmap_ava, _ = read_labelmap("D:/Data/AVA/annotations/ava_action_list_v2.2.pbtxt")
    labelmap_df2, _ = read_labelmap("D:/Data/DeepFashion2/df2_list.pbtxt")

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
                                             augment=False, hyp=hyp, rect=False, image_weights=opt.image_weights, #augment is always False
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
                                                augment=False, hyp=opt.hyp, rect=False, image_weights=opt.image_weights, #augment is always False
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
    accumulation_steps = 8
    optimizer = create_optimizer_v2(model.parameters(), opt='adam', lr=hyp['lr0'], momentum=hyp['momentum'], weight_decay=hyp['weight_decay'])
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=hyp['lrmax'], total_steps=int(epochs * num_batch / accumulation_steps), div_factor=int(hyp['lrmax'] / hyp['lr0']))
    
    # EMA
    # ema = ModelEMA(model)

    # 8. Resume
    start_epoch, best_fitness = 0, 1e+9
    if pretrained:
        # Optimizer
        if ckpt['optimizer'] is not None:
            optimizer.load_state_dict(ckpt['optimizer'])
            best_fitness = ckpt['best_fitness']
        # EMA
        # if ema and ckpt.get('ema'):
        #     ema.ema.load_state_dict(ckpt['ema'].float().state_dict())
        #     ema.updates = ckpt['updates']
        
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
    scaler = amp.GradScaler(enabled=cuda)
    logger.info(f'Using {dataloader.num_workers} dataloader workers\n'
                f'Logging results to {save_dir}\n'
                f'Starting training for {epochs} epochs...')
    LOSS = ComputeLoss(detector_head=model.head_bbox, hyp=hyp)
    WLOSS = WeightedMultiTaskLoss(num_tasks=4)
    
    # Start epoch ------------------------------------------------------------------------------------------------------
    for epoch in range(start_epoch, epochs):  
        logger.info(('\n' + '%10s' * 7) % ('gpu_mem', 'box', 'obj', 'act', 'clo', 'total', 'lr'))
        model.train()
        # model._freeze_modules() # No freezing backbone
        optimizer.zero_grad()
        pbar = enumerate(dataloader)
        pbar = tqdm(pbar, total=num_batch)  # progress bar
        mloss_1 = torch.zeros(1, device= 'cpu') # mean losses for item1
        mloss_2 = torch.zeros(3, device= 'cpu') # mean losses for item2
        mloss = torch.zeros(4, device='cpu')  # mean losses
        mtotal_loss = torch.zeros(1, device='cpu') # mean total_loss (sum of losses)
        
        # Start batch ----------------------------------------------------------------------------------------------------
        i_1, i_2 = 0, 0
        for i, (item1, item2) in pbar:
            # Batch-01. Input data
            if random.random() < 0.5:
                select = 'DF2'
                i_1 += 1
                '''
                Explanation of variables in item1_batch:
                - imgs (torch.uint8): Image data with shape [B, 3, H, W] which has the scale of 0~255
                - labels (torch.float32): Label data with shape [num, 6], where num is the number of objects in the image.
                                            Each row in the labels contains: [Batch_num, class_num, x, y, h, w].
                - paths (tuple[str]): List of image paths with length B.
                - _shapes (tuple[tuple]): List of tuples, each containing (h0, w0), ((h / h0, w / w0), pad).
                - pseudo_feature_DF2 (torch.float32): Features data with shape [B, 15, 7, 7], representing 3 (num_anchor) x 5 (bbox xywh + confidence score).
                '''
                imgs, labels, paths, _shapes, pseudo_feature_DF2 = item1
                imgs = imgs.to(device, non_blocking=True).float() / 255.0  # uint8 to float32, 0-255 to 0.0-1.0
                model_input = imgs.unsqueeze(2).repeat((1, 1, opt.DATA.NUM_FRAMES, 1, 1)) # shape [B, 3, T, H, W]
            else:
                select = 'AVA'
                i_2 += 1
                '''
                Explanation of variables in item2_batch:
                - clips (torch.float32): Video clips data with shape [B, 3, T, H, W] which has the scale of 0~1
                - cls (np.array, dtype('float32')): Array of class data with shape [B, 50, 80].
                                                    It contains up to 50 labels (from the beginning to the end).
                - boxes (np.array, dtype('float32')): Array of bounding box data with shape [B, 50, 4].
                - pseudo_feature_AVA (np.array, dtype('float16')): Array of feature data with shape [B, 5(na), 7, 7, 13(5+8)].
                '''
                clips, cls, boxes, pseudo_feature_AVA = item2
                clips = clips.to(device, non_blocking=True)
                model_input = clips # shape [B, 3, T, H, W]
            
            # Batch-02. Forward
            with amp.autocast(enabled=cuda):
                out_bboxs, out_clos, out_acts = model(model_input)
                
                out_bbox_infer, out_bbox_features = out_bboxs[0], out_bboxs[1]
                out_clo_infer, out_clo_features = out_clos[0], out_clos[1]
                out_act_infer, out_act_features = out_acts[0], out_acts[1]

                #TODO: Define the loss function
                if select == 'DF2':
                    _dtype = out_bbox_features[0].dtype
                    pseudo_feature_DF2 = torch.tensor(pseudo_feature_DF2, dtype= _dtype, device=device)
                    pseudo_bbox = pseudo_feature_DF2.view(batch_size, na, 5, 7, 7).permute(0, 1, 3, 4, 2) # [B, na, 7, 7, (4+1)]
                    
                    _lbox, _lobj, lclo, feature_mse_loss = LOSS.forward_df2(
                        p_clo=out_clo_features, 
                        p_bbox=out_bbox_features,
                        pseudo_bbox=pseudo_bbox,
                        targets=labels)
                    
                    total_loss = lclo + feature_mse_loss
                
                elif select == 'AVA':
                    _dtype = out_clo_features[0].dtype
                    pseudo_feature_AVA = torch.tensor(pseudo_feature_AVA, dtype= _dtype, device=device)
                    pseudo_cloth = pseudo_feature_AVA[..., 5:]
                    
                    lbox, lobj, lact, feature_mse_loss = LOSS.forward_ava(
                        p_act=out_act_features, 
                        p_bbox=out_bbox_features, 
                        t_cls=cls, 
                        t_bbox=boxes,
                        p_clo= out_clo_features, 
                        pseudo_cloth= pseudo_cloth)
                    
                    total_loss = lbox + lobj + lact + feature_mse_loss

            # Batch-03. Backward
            scaler.scale(total_loss).backward()
            
            if (i+1) % accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()
                # if ema: ema.update(model)

            # Batch-04. Print
            if select == 'DF2':
                loss_item = torch.tensor([lclo], device='cpu')
                if torch.all(torch.isfinite(loss_item)):
                    mloss_1 = (mloss_1 * i_1 + loss_item) / (i_1 + 1)  # update mean losses
                    mloss[3] = mloss_1
                    
            elif select == 'AVA':
                loss_item = torch.tensor([lbox, lobj, lact], device='cpu')
                if torch.all(torch.isfinite(loss_item)):
                    mloss_2 = (mloss_2 * i_2 + loss_item) / (i_2 + 1)  # update mean losses
                    mloss[0:3] = mloss_2
            
            mtotal_loss = (mtotal_loss * i + total_loss.detach().cpu()) / (i+1) # update mean total_loss
            
            # verbose's string
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
                tags = ['train/box_loss', 'train/obj_loss', 'train/act_loss', 'train/clo_loss', 'train/lr']
                for x, tag in zip(list(mloss) + lr, tags):
                    wandb_logger.log({tag: x})
                    
            # Batch-05. Plot
            if (plots) and (cur_step % opt.log_step == 0):
                plot_i = (cur_step // opt.log_step) % 4
                f_clo = save_dir / f'train_batch_clo{plot_i}.jpg'  # filename
                f_act = save_dir / f'train_batch_act{plot_i}.jpg'  # filename
                
                keyframes = model_input[:, :, -1, :, :] # keyframes for plot
                keyframes = un_normalized_images(keyframes)
                
                preds_clo = torch.cat((out_bbox_infer, out_clo_infer), dim=2)
                preds_clo = non_max_suppression(preds_clo, conf_thres=0.3, iou_thres=0.5)
                Thread(target=plot_batch_image_from_preds, args=(keyframes.copy(), preds_clo, str(f_clo), labelmap_df2), daemon=True).start()
                
                preds_act = torch.cat((out_bbox_infer, out_act_infer), dim=2)
                preds_act = non_max_suppression(preds_act, conf_thres=0.5, iou_thres=0.5)
                Thread(target=plot_batch_image_from_preds, args=(keyframes.copy(), preds_act,str(f_act), labelmap_ava), daemon=True).start()

            elif plots and i == 5 and wandb_logger.wandb:
                wandb_logger.log({"Mosaics": [wandb_logger.wandb.Image(str(x), caption=x.name) for x in
                                                save_dir.glob('train*.jpg') if x.exists()]})

            # End batch ----------------------------------------------------------------------------------------------------

        # End epoch --------------------------------------------------------------------------------------------------------
    
        # Start write ------------------------------------------------------------------------------------------------------
        # ema.update_attr(model, include=['yaml', 'nc', 'hyp', 'gr', 'names', 'stride', 'class_weights'])
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
                    # 'ema': deepcopy(ema.ema).half(),
                    # 'updates': ema.updates,
                    'optimizer': optimizer.state_dict(),
                    'wandb_id': wandb_logger.wandb_run.id if wandb_logger.wandb else None}

            # Save last, best and delete
            last = wdir / f'epoch{str(epoch)}.pt'
            torch.save(ckpt, last)
            if best_fitness == fi:
                torch.save(ckpt, best)
            del ckpt

        # End write --------------------------------------------------------------------------------------------------------
    # End training -----------------------------------------------------------------------------------------------------

if __name__ == '__main__':
    # For overwriting parameters of 'cfg/deepfashion2.yaml'
    # You do not have to set the parameters below, if you set the correct paramters in 'cfg/deepfashion2.yaml'
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, help='initial weights path')
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--batch-size', type=int, help='total batch size for all GPUs')
    parser.add_argument('--img-size', nargs='+', type=int, help='[train, test] image sizes')
    parser.add_argument('--rect', action='store_true', help='rectangular training')
    parser.add_argument('--resume', nargs='?', help='resume most recent training')
    parser.add_argument('--nosave', action='store_true', help='only save final checkpoint')
    parser.add_argument('--notest', action='store_true', help='only test final epoch')
    parser.add_argument('--noautoanchor', action='store_true', help='disable autoanchor check')
    parser.add_argument('--cache-images', action='store_true', help='cache images for faster training')
    parser.add_argument('--image-weights', action='store_true', help='use weighted image selection for training')
    parser.add_argument('--device', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--single-cls', action='store_true', help='train multi-class data as single-class')
    parser.add_argument('--sync-bn', action='store_true', help='use SyncBatchNorm, only available in DDP mode')
    parser.add_argument('--local_rank', type=int, help='DDP parameter, do not modify')
    parser.add_argument('--workers', type=int, help='maximum number of dataloader workers')
    parser.add_argument('--project', default='runs/train', help='save to project/name')
    parser.add_argument('--entity', default=None, help='W&B entity')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--bbox_interval', type=int, default=1, help='Set bounding-box image logging interval for W&B')
    parser.add_argument('--save_period', type=int, default=-1, help='Log model after every "save_period" epoch')
    parser.add_argument('--v5-metric', action='store_true', help='assume maximum recall as 1.0 in AP calculation')
    parser.add_argument('--log_step', action='store_true', default= 100, help='after every logging_step record the performance to the wandb')
    opt_ = parser.parse_args()
    
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
    opt.merge(opt_) # overwrite
    
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