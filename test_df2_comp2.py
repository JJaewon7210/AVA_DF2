import argparse
import json
import os
from pathlib import Path
from threading import Thread
from itertools import islice

import numpy as np
import torch
import yaml
from tqdm import tqdm

from utils.general import coco80_to_coco91_class, check_img_size, non_max_suppression, \
    scale_coords, xyxy2xywh, xywh2xyxy, set_logging, increment_path, ConfigObject, box_iou_only_box1
from utils.metrics import ap_per_class, ConfusionMatrix
from utils.plots import plot_images, read_labelmap, un_normalized_images, plot_batch_image_from_preds
from utils.torch_utils import select_device, time_synchronized, TracedModel
from datasets.yolo_datasets import InfiniteDataLoader, LoadImagesAndLabels

def test_df2(
             opt,
         weights=None,
         batch_size=32,
         imgsz=640,
         conf_thres=0.01,
         iou_thres=0.6,  # for NMS
         save_json=False,
         single_cls=False,
         augment=False,
         verbose=False,
         model=None,
         dataloader=None,
         save_dir=Path(''),  # for saving images
         save_txt=False,  # for auto-labelling
         save_hybrid=False,  # for hybrid auto-labelling
         save_conf=False,  # save auto-label confidences
         plots=True,
         wandb_logger=None,
         compute_loss=None,
         half_precision=True,
         trace=False,
         is_coco=False,
         v5_metric=False):
    # Initialize/load model and set device
    training = model is not None
    if training:  # called by train.py
        device = next(model.parameters()).device  # get model device

    else:  # called directly
        set_logging()
        device = select_device(opt.device, batch_size=batch_size)

        # Directories
        save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
        (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

        # Load model
        model = torch.load(weights, map_location=device)  # load FP32 model
        gs = 32  # grid size (max stride)
        imgsz = check_img_size(imgsz, s=gs)  # check img_size
        
        if trace:
            model = TracedModel(model, device, imgsz)
            
    # Load model
    if type(model) == dict:
        if 'ema' in model:
            model = model['ema']
        else:
            model = model['model']
    # Half
    half = device.type != 'cpu' and half_precision  # half precision only supported on CUDA
    if half:
        model.half()

    # Configure
    model.eval()
    nc = 13
    iouv = torch.linspace(0.5, 0.95, 10).to(device)  # iou vector for mAP@0.5:0.95
    niou = iouv.numel()
    labelmap_df2, _ = read_labelmap("D:/Data/DeepFashion2/df2_list.pbtxt")
    
    # Logging
    log_imgs = 0
    if wandb_logger and wandb_logger.wandb:
        log_imgs = min(wandb_logger.log_imgs, 100)
    # Dataloader
    if not training:
        if device.type != 'cpu':
            model(torch.zeros(1, 3, 16, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once

        testset_df2 = LoadImagesAndLabels(path=opt.val, img_size=imgsz, batch_size=opt.batch_size_test, 
                                                augment=False, hyp=opt.hyp, rect=False, image_weights=opt.image_weights,
                                                cache_images=opt.cache_images, single_cls=opt.single_cls, 
                                                stride=32, pad=0.0, prefix='val: ')

        loader = torch.utils.data.DataLoader if opt.image_weights else InfiniteDataLoader
        dataloader = loader(testset_df2, batch_size=opt.batch_size_test, num_workers=opt.workers, collate_fn=LoadImagesAndLabels.collate_fn)
        

    if v5_metric:
        print("Testing with YOLOv5 AP metric...")
    
    seen = 0
    confusion_matrix = ConfusionMatrix(nc=nc)
    names_str = opt.names
    names = {k: v for k, v in enumerate(names_str)}
    coco91class = coco80_to_coco91_class()
    s = ('%20s' + '%12s' * 6) % ('Class', 'Images', 'Labels', 'P', 'R', 'mAP@.5', 'mAP@.5:.95')
    p, r, f1, mp, mr, map50, map, t0, t1 = 0., 0., 0., 0., 0., 0., 0., 0., 0.
    loss = torch.zeros(3, device=device)
    jdict, stats, ap, ap_class, wandb_images = [], [], [], [], []
    for batch_i, (img, targets, paths, shapes) in enumerate(tqdm(dataloader, desc=s)):
    # for batch_i, (img, targets, paths, shapes) in enumerate(tqdm(islice(dataloader, 15077, None), desc=s)):

        # Step1: Load txt file. txt file contains several lines that denotes [cls_idx, xc, yc, h, w, conf_score]. xc, yc, h, w has the value between 0 from 1.
        # Be careful that there is the case where txt file is not exist.
        # txt path: save_dir / 'labels' / (path.stem + '.txt')
        txt_paths = ['C:/CNN/AVA_DF2/runs/test/YOWO/labels_refined' + '/' + (Path(path).stem + '.txt') for path in paths]
        txt_contents = []
        for txt_path in txt_paths:
            if os.path.exists(txt_path):
                with open(txt_path, 'r') as f:
                    lines = f.readlines()
                    values= []
                    for line in lines:
                        values.append([float(x) for x in line.strip().split()])
                    txt_contents.append(values)
            else:
                txt_contents.append([])
                
        # Step 2: Crop the images based on the loaded txt file
        cropped_images = []
        human_bboxs = []
        
        for i in range(len(img)):
            img_i = img[i]  # Get the i-th image
            txt_i = txt_contents[i]  # Get the corresponding txt content
            
            if txt_i:
                # If there is txt content for this image
                cropped_img_i = []
                human_bbox_i = []
                for cls_idx, xc, yc, h, w, conf_score in txt_i:
                    # Crop the image based on xc, yc, h, and w
                    xc =xc * img_i.shape[2]
                    yc =yc * img_i.shape[1]
                    h =h * img_i.shape[2]
                    w =w * img_i.shape[1]
                    
                    x1 = max(int(xc - w//2 - w*0.1),0)
                    x2 = min(int(xc + w//2 + w*0.1),img_i.shape[2])
                    y1 = max(int(yc - h//2 - h*0.1),0)
                    y2 = min(int(yc + h//2 + h*0.1),img_i.shape[1])
                    # xc, yc, h, w = map(int, [xc * img_i.shape[2], yc * img_i.shape[1], h * img_i.shape[2], w * img_i.shape[1]])
                    cropped_img = img_i[:, x1:x2,y1:y2]
                    black_canvas = torch.zeros_like(img_i)
                    black_canvas[:, x1:x2,y1:y2] = cropped_img
                    cropped_img_i.append(black_canvas)
                    human_bbox_i.append([int(xc - w//2), int(yc - h//2), int(xc + w//2), int(yc + h//2)]) # x1, y1, x2, y2
                    
                cropped_images.append(torch.stack(cropped_img_i, dim=0))
                human_bboxs.append(human_bbox_i)
            else:
                # If there is no txt content for this image, use the original image
                # NOTE: It only has single batch
                cropped_images.append(torch.zeros_like(img_i).unsqueeze(0))
                human_bboxs.append([[0,0,0,0]]) # x1=y1=x2=y2=0
        
        
        # Step 3: Make multiple inputs based on the cropped images
        img = torch.cat(cropped_images, dim=0)
        img = img.to(device, non_blocking=True)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        targets = targets.to(device)
        imgs_duplicated = img.unsqueeze(2).repeat((1, 1, opt.DATA.NUM_FRAMES, 1, 1))
        keyframes = imgs_duplicated[:, :, -1, :, :]  # keyframes
        nb, _c, _t, height, width = imgs_duplicated.shape  # batch size, channels, T, height, width
        

        with torch.no_grad():
            # Run model
            t = time_synchronized()
            out_bboxs, out_clos = model(imgs_duplicated)
            
            out_bbox_infer, out_bbox_features = out_bboxs[0], out_bboxs[1]
            out_clo_infer, out_clo_features = out_clos[0], out_clos[1]
            # out_act_infer, out_act_features = out_acts[0], out_acts[1]
            out_pred = torch.cat((out_bbox_infer, out_clo_infer), dim=2)
            t0 += time_synchronized() - t


            # Run NMS
            lb = [targets[targets[:, 0] == i, 1:] for i in range(nb)] if save_hybrid else []  # for autolabelling
            t = time_synchronized()
            out = non_max_suppression(out_pred, conf_thres=conf_thres, iou_thres=iou_thres, labels=lb, multi_label=True)
            t1 += time_synchronized() - t

        # Statistics per image (human)
        for si, pred in enumerate(out):
            # pred shape [n, 6] or 0
            # pred has [xyxy, conf, cls]
            
            labels = targets[targets[:, 0] == si, 1:]
            nl = len(labels)
            tcls = labels[:, 0].tolist() if nl else []  # target class
            path = Path(paths[0])
            seen += 1

            if len(pred) == 0:
                if nl:
                    stats.append((torch.zeros(0, niou, dtype=torch.bool), torch.Tensor(), torch.Tensor(), tcls))
                continue
            
            x1 = human_bboxs[0][si][0] /224 # Only single batch
            y1 = human_bboxs[0][si][1] /224
            x2 = human_bboxs[0][si][2] /224
            y2 = human_bboxs[0][si][3] /224

            pred[:, 0] = x1
            pred[:, 1] = y1
            pred[:, 2] = x2
            pred[:, 3] = y2
            
            # Predictions
            predn = pred.clone()
            # scale_coords(img[si].shape[1:], predn[:, :4], shapes[si][0], shapes[si][1])  # native-space pred

            # Append to text file
            if save_txt:
                gn = torch.tensor(shapes[si][0])[[1, 0, 1, 0]]  # normalization gain whwh
                for *xyxy, conf, cls in predn.tolist():
                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                    line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                    with open(save_dir / 'labels' / (path.stem + '.txt'), 'a') as f:
                        f.write(('%g ' * len(line)).rstrip() % line + '\n')

            # W&B logging - Media Panel Plots
            if len(wandb_images) < log_imgs and wandb_logger.current_epoch > 0:  # Check for test operation
                if wandb_logger.current_epoch % wandb_logger.bbox_interval == 0:
                    box_data = [{"position": {"minX": xyxy[0], "minY": xyxy[1], "maxX": xyxy[2], "maxY": xyxy[3]},
                                 "class_id": int(cls),
                                 "box_caption": "%s %.3f" % (names[cls], conf),
                                 "scores": {"class_score": conf},
                                 "domain": "pixel"} for *xyxy, conf, cls in pred.tolist()]
                    boxes = {"predictions": {"box_data": box_data, "class_labels": names}}  # inference-space
                    wandb_images.append(wandb_logger.wandb.Image(img[si], boxes=boxes, caption=path.name))
            wandb_logger.log_training_progress(predn, path, names) if wandb_logger and wandb_logger.wandb_run else None

            # Append to pycocotools JSON dictionary
            if save_json:
                # [{"image_id": 42, "category_id": 18, "bbox": [258.15, 41.29, 348.26, 243.78], "score": 0.236}, ...
                image_id = int(path.stem) if path.stem.isnumeric() else path.stem
                box = xyxy2xywh(predn[:, :4])  # xywh
                box[:, :2] -= box[:, 2:] / 2  # xy center to top-left corner
                for p, b in zip(pred.tolist(), box.tolist()):
                    jdict.append({'image_id': image_id,
                                  'category_id': coco91class[int(p[5])] if is_coco else int(p[5]),
                                  'bbox': [round(x, 3) for x in b],
                                  'score': round(p[4], 5)})

            # Assign all predictions as incorrect
            correct = torch.zeros(pred.shape[0], niou, dtype=torch.bool, device=device)
            if nl:
                detected = []  # target indices
                tcls_tensor = labels[:, 0]

                # target boxes
                tbox = xywh2xyxy(labels[:, 1:5])
                # scale_coords(img[si].shape[1:], tbox, shapes[si][0], shapes[si][1])  # native-space labels
                if plots:
                    confusion_matrix.process_batch(predn, torch.cat((labels[:, 0:1], tbox), 1), only_box1=True)

                # Per target class
                for cls in torch.unique(tcls_tensor):
                    ti = (cls == tcls_tensor).nonzero(as_tuple=False).view(-1)  # prediction indices
                    pi = (cls == pred[:, 5]).nonzero(as_tuple=False).view(-1)  # target indices

                    # Search for detections
                    if pi.shape[0]:
                        # Prediction to target ious
                        # ious, i = box_iou(predn[pi, :4], tbox[ti]).max(1) 
                        ious, i = box_iou_only_box1(predn[pi, :4], tbox[ti], standard='box2').max(1)  # best ious, indices of the target boxes that have the highest IoU with each prediction.
                        # Append detections
                        detected_set = set()
                        for j in (ious > iouv[0]).nonzero(as_tuple=False):
                            d = ti[i[j]]  # detected target
                            if d.item() not in detected_set:
                                detected_set.add(d.item())
                                detected.append(d)
                                correct[pi[j]] = ious[j] > iouv  # iou_thres is 1xn
                                if len(detected) == nl:  # all targets already located in image
                                    break

            # Append statistics (correct, conf, pcls, tcls)
            stats.append((correct.cpu(), pred[:, 4].cpu(), pred[:, 5].cpu(), tcls))

        # Plot images
        if plots and batch_i < 30:
            f = save_dir / f'test_df2_batch{batch_i}_labels.jpg'  # labels
            targets[:, 2:] *= torch.Tensor([width, height, width, height]).to(device)  # to pixels
            Thread(target=plot_images, args=(img, targets, None, f, names), daemon=True).start()
            
            f = save_dir / f'test_df2_batch{batch_i}_pred.jpg'  # predictions
            keyframes = un_normalized_images(keyframes)
            outs = non_max_suppression(out_pred, conf_thres=0.3, iou_thres=0.5, cls_thres=0.25)
            Thread(target=plot_batch_image_from_preds, args=(keyframes.copy(), outs, str(f), labelmap_df2, (224,224), False), daemon=True).start()

    # Compute statistics
    stats = [np.concatenate(x, 0) for x in zip(*stats)]  # to numpy
    if len(stats) and stats[0].any():
        p, r, ap, f1, ap_class = ap_per_class(*stats, plot=plots, v5_metric=v5_metric, save_dir=save_dir, names=names)
        ap50, ap = ap[:, 0], ap.mean(1)  # AP@0.5, AP@0.5:0.95
        mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
        nt = np.bincount(stats[3].astype(np.int64), minlength=nc)  # number of targets per class
    else:
        nt = torch.zeros(1)

    # Print results
    pf = '%20s' + '%12i' * 2 + '%12.3g' * 4  # print format
    print(pf % ('all', seen, nt.sum(), mp, mr, map50, map))

    # Print results per class
    if (verbose or (nc < 50 and not training)) and nc > 1 and len(stats):
        for i, c in enumerate(ap_class):
            print(pf % (names[c], seen, nt[c], p[i], r[i], ap50[i], ap[i]))

    # Print speeds
    t = tuple(x / seen * 1E3 for x in (t0, t1, t0 + t1)) + (imgsz, imgsz, batch_size)  # tuple
    if not training:
        print('Speed: %.1f/%.1f/%.1f ms inference/NMS/total per %gx%g image at batch-size %g' % t)

    # Plots
    if plots:
        confusion_matrix.plot(save_dir=save_dir, names=list(names.values()))
        if wandb_logger and wandb_logger.wandb:
            val_batches = [wandb_logger.wandb.Image(str(f), caption=f.name) for f in sorted(save_dir.glob('test_df2*.jpg'))]
            wandb_logger.log({"Validation": val_batches})
    if wandb_images:
        wandb_logger.log({"Bounding Box Debugger/Images_df2": wandb_images})

    # Save JSON
    if save_json and len(jdict):
        w = Path(weights[0] if isinstance(weights, list) else weights).stem if weights is not None else ''  # weights
        anno_json = './coco/annotations/instances_val2017.json'  # annotations json
        pred_json = str(save_dir / f"{w}_predictions.json")  # predictions json
        print('\nEvaluating pycocotools mAP... saving %s...' % pred_json)
        with open(pred_json, 'w') as f:
            json.dump(jdict, f)

        try:  # https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocoEvalDemo.ipynb
            from pycocotools.coco import COCO
            from pycocotools.cocoeval import COCOeval

            anno = COCO(anno_json)  # init annotations api
            pred = anno.loadRes(pred_json)  # init predictions api
            eval = COCOeval(anno, pred, 'bbox')
            if is_coco:
                eval.params.imgIds = [int(Path(x).stem) for x in dataloader.dataset.img_files]  # image IDs to evaluate
            eval.evaluate()
            eval.accumulate()
            eval.summarize()
            map, map50 = eval.stats[:2]  # update results (mAP@0.5:0.95, mAP@0.5)
        except Exception as e:
            print(f'pycocotools unable to run: {e}')

    # Return results
    model.float()  # for training
    if not training:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        print(f"Results saved to {save_dir}{s}")
    maps = np.zeros(nc) + map
    for i, c in enumerate(ap_class):
        maps[c] = ap[i]
    return (mp, mr, map50, map, *(loss.cpu() / len(dataloader)).tolist()), maps, t


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='test.py')
    parser.add_argument('--weights', type=str, default='runs/train/augment_True_only_df2/weights/epoch48.pt', help='model.pt path(s)')
    parser.add_argument('--batch-size-test', type=int, default=1, help='size of each image batch')
    parser.add_argument('--img-size', type=int, default=224, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.3, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
    parser.add_argument('--task', default='val', help='train, val, test, speed or study')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--single-cls', action='store_true', help='treat as single-class dataset')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--verbose', action='store_true', help='report mAP by class')
    parser.add_argument('--save-txt', default= False, action='store_true', help='save results to *.txt')
    parser.add_argument('--save-hybrid', action='store_true', help='save label+prediction hybrid results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-json', action='store_true', help='save a cocoapi-compatible JSON results file')
    parser.add_argument('--project', default='runs/test', help='save to project/name')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--no-trace', action='store_true', help='don`t trace model')
    parser.add_argument('--v5-metric', action='store_true', help='assume maximum recall as 1.0 in AP calculation')
    opt_ = parser.parse_args()
    
    
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
    
    opt = ConfigObject({})
    opt.hyp = hyp
    opt.merge(opt_df2)
    opt.merge(opt_ava)
    opt.merge(opt_model)
    opt.merge(opt_) # overwrite
    
    #check_requirements()

    if opt.task in ('val', 'test'):  # run normally
        test_df2(opt,
             opt.weights,
             opt.batch_size_test,
             opt.img_size,
             opt.conf_thres,
             opt.iou_thres,
             opt.save_json,
             opt.single_cls,
             opt.augment,
             opt.verbose,
             save_txt=opt.save_txt | opt.save_hybrid,
             save_hybrid=opt.save_hybrid,
             save_conf=opt.save_conf,
             v5_metric=opt.v5_metric
             )

    elif opt.task == 'speed':  # speed benchmarks
        for w in opt.weights:
            test_df2(opt, 
                     w, 
                     opt.batch_size_test, 
                     opt.img_size, 
                     0.25, 
                     0.45, 
                     save_json=False, 
                     plots=False, 
                     v5_metric=opt.v5_metric)