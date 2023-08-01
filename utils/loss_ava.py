import time
import json
import torch
import torch.nn.functional as F

import math
import torch.nn as nn
from torch.autograd import Variable
from builtins import range as xrange
from utils.general import box_iou, xywh2xyxy
import numpy as np

def extract_bounding_boxes(A):
    """
    This function extracts non-zero bounding boxes from a batched tensor or array.

    Parameters:
    A (numpy.ndarray or torch.Tensor): A 3D tensor/array of shape [Batch_size, 50, 4].
                                       Each batch can contain up to 50 bounding boxes (x, y, w, h),
                                       where zero bounding boxes are represented by [0, 0, 0, 0].

    Returns:
    B (numpy.ndarray or torch.Tensor): A 2D tensor/array of shape [num_boxes, 5],
                                       where num_boxes is the total number of non-zero bounding boxes in A,
                                       and the 5 columns represent [batch_num, x, y, w, h] for each bounding box.

    Note:
    This function assumes that all zero bounding boxes are represented by [0, 0, 0, 0].
    """

    B = []

    if isinstance(A, np.ndarray):
        for batch_num, batch in enumerate(A):
            for bbox in batch:
                if np.any(bbox != 0):  # if the bbox is not all zeros
                    B.append([batch_num] + list(bbox))
        B = np.array(B)

    elif torch.is_tensor(A):
        for batch_num in range(A.size(0)):
            for bbox_num in range(A.size(1)):
                if torch.any(A[batch_num, bbox_num] != 0):  # if the bbox is not all zeros
                    B.append([batch_num] + A[batch_num, bbox_num].tolist())
        B = torch.tensor(B, device=A.device, dtype=A.dtype)  # make sure B has the same dtype and device as A

    else:
        print("Unsupported input type.")

    return B
    
def convert_one_hot_to_batch_class(A):
    """
    This function converts a one-hot encoded array or tensor to a 2D tensor/array, where the second dimension represents [batch_num, class_num].

    Parameters:
    A (numpy.ndarray or torch.Tensor): A 3D one-hot encoded array/tensor of shape [Batch_size, 50, 80].
                                      Each batch can contain up to 50 labels, each one-hot encoded with 80 classes.

    Returns:
    B (numpy.ndarray or torch.Tensor): A 2D tensor/array of shape [num_obj, 2], where num_obj is the total number of non-zero labels in A,
                                       and the 2 columns represent [Batch_num, class_num] for each label.

    Note:
    This function assumes that A has only one '1' per label. If a label can have multiple '1's, the function needs to be adjusted accordingly.
    """
    B = []

    if isinstance(A, np.ndarray):
        for batch_num, batch in enumerate(A):
            for label_num, label in enumerate(batch):
                if np.sum(label) != 0:  # if the label is not all zeros
                    class_num = np.argmax(label)  # get the index of the 1 in one-hot encoded class
                    B.append([batch_num, class_num])
        B = np.array(B)

    elif torch.is_tensor(A):
        for batch_num in range(A.size(0)):
            for label_num in range(A.size(1)):
                if torch.sum(A[batch_num, label_num]) != 0:  # if the label is not all zeros
                    class_num = torch.argmax(A[batch_num, label_num])  # get the index of the 1 in one-hot encoded class
                    B.append([batch_num, class_num.item()])
        B = torch.tensor(B, device=A.device, dtype=A.dtype)  # make sure B has the same dtype and device as A

    else:
        print("Unsupported input type.")

    return B

def bbox_ious(boxes1, boxes2, x1y1x2y2=True):
    if x1y1x2y2:
        mx = torch.min(boxes1[0], boxes2[0])
        Mx = torch.max(boxes1[2], boxes2[2])
        my = torch.min(boxes1[1], boxes2[1])
        My = torch.max(boxes1[3], boxes2[3])
        w1 = boxes1[2] - boxes1[0]
        h1 = boxes1[3] - boxes1[1]
        w2 = boxes2[2] - boxes2[0]
        h2 = boxes2[3] - boxes2[1]
    else:
        mx = torch.min(boxes1[0]-boxes1[2]/2.0, boxes2[0]-boxes2[2]/2.0)
        Mx = torch.max(boxes1[0]+boxes1[2]/2.0, boxes2[0]+boxes2[2]/2.0)
        my = torch.min(boxes1[1]-boxes1[3]/2.0, boxes2[1]-boxes2[3]/2.0)
        My = torch.max(boxes1[1]+boxes1[3]/2.0, boxes2[1]+boxes2[3]/2.0)
        w1 = boxes1[2]
        h1 = boxes1[3]
        w2 = boxes2[2]
        h2 = boxes2[3]
    uw = Mx - mx
    uh = My - my
    cw = w1 + w2 - uw
    ch = h1 + h2 - uh
    mask = ((cw <= 0) + (ch <= 0) > 0)
    area1 = w1 * h1
    area2 = w2 * h2
    carea = cw * ch
    carea[mask] = 0
    uarea = area1 + area2 - carea
    return carea/uarea

def bbox_iou(box1, box2, x1y1x2y2=True):
    if x1y1x2y2:
        mx = min(box1[0], box2[0])
        Mx = max(box1[2], box2[2])
        my = min(box1[1], box2[1])
        My = max(box1[3], box2[3])
        w1 = box1[2] - box1[0]
        h1 = box1[3] - box1[1]
        w2 = box2[2] - box2[0]
        h2 = box2[3] - box2[1]
    else:
        mx = min(float(box1[0]-box1[2]/2.0), float(box2[0]-box2[2]/2.0))
        Mx = max(float(box1[0]+box1[2]/2.0), float(box2[0]+box2[2]/2.0))
        my = min(float(box1[1]-box1[3]/2.0), float(box2[1]-box2[3]/2.0))
        My = max(float(box1[1]+box1[3]/2.0), float(box2[1]+box2[3]/2.0))
        w1 = box1[2]
        h1 = box1[3]
        w2 = box2[2]
        h2 = box2[3]
    uw = Mx - mx
    uh = My - my
    cw = w1 + w2 - uw
    ch = h1 + h2 - uh
    carea = 0
    if cw <= 0 or ch <= 0:
        return 0.0

    area1 = w1 * h1
    area2 = w2 * h2
    carea = cw * ch
    uarea = area1 + area2 - carea
    return carea/uarea

def convert2cpu(gpu_matrix):
    return torch.FloatTensor(gpu_matrix.size()).copy_(gpu_matrix)

########################### AVA ###############################
###############################################################
class SigmoidBin(nn.Module):
    stride = None  # strides computed during build
    export = False  # onnx export

    def __init__(self, bin_count=10, min=0.0, max=1.0, reg_scale = 2.0, use_loss_regression=True, use_fw_regression=True, BCE_weight=1.0, smooth_eps=0.0):
        super(SigmoidBin, self).__init__()
        
        self.bin_count = bin_count
        self.length = bin_count + 1
        self.min = min
        self.max = max
        self.scale = float(max - min)
        self.shift = self.scale / 2.0

        self.use_loss_regression = use_loss_regression
        self.use_fw_regression = use_fw_regression
        self.reg_scale = reg_scale
        self.BCE_weight = BCE_weight

        start = min + (self.scale/2.0) / self.bin_count
        end = max - (self.scale/2.0) / self.bin_count
        step = self.scale / self.bin_count
        self.step = step
        #print(f" start = {start}, end = {end}, step = {step} ")

        bins = torch.range(start, end + 0.0001, step).float() 
        self.register_buffer('bins', bins) 
               

        self.cp = 1.0 - 0.5 * smooth_eps
        self.cn = 0.5 * smooth_eps

        self.BCEbins = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([BCE_weight]))
        self.MSELoss = nn.MSELoss()

    def get_length(self):
        return self.length

    def forward(self, pred):
        assert pred.shape[-1] == self.length, 'pred.shape[-1]=%d is not equal to self.length=%d' % (pred.shape[-1], self.length)

        pred_reg = (pred[..., 0] * self.reg_scale - self.reg_scale/2.0) * self.step
        pred_bin = pred[..., 1:(1+self.bin_count)]

        _, bin_idx = torch.max(pred_bin, dim=-1)
        bin_bias = self.bins[bin_idx]

        if self.use_fw_regression:
            result = pred_reg + bin_bias
        else:
            result = bin_bias
        result = result.clamp(min=self.min, max=self.max)

        return result


    def training_loss(self, pred, target):
        assert pred.shape[-1] == self.length, 'pred.shape[-1]=%d is not equal to self.length=%d' % (pred.shape[-1], self.length)
        assert pred.shape[0] == target.shape[0], 'pred.shape=%d is not equal to the target.shape=%d' % (pred.shape[0], target.shape[0])
        device = pred.device

        pred_reg = (pred[..., 0].sigmoid() * self.reg_scale - self.reg_scale/2.0) * self.step
        pred_bin = pred[..., 1:(1+self.bin_count)]

        diff_bin_target = torch.abs(target[..., None] - self.bins)
        _, bin_idx = torch.min(diff_bin_target, dim=-1)
    
        bin_bias = self.bins[bin_idx]
        bin_bias.requires_grad = False
        result = pred_reg + bin_bias

        target_bins = torch.full_like(pred_bin, self.cn, device=device)  # targets
        n = pred.shape[0] 
        target_bins[range(n), bin_idx] = self.cp

        loss_bin = self.BCEbins(pred_bin, target_bins) # BCE

        if self.use_loss_regression:
            loss_regression = self.MSELoss(result, target)  # MSE        
            loss = loss_bin + loss_regression
        else:
            loss = loss_bin

        out_result = result.clamp(min=self.min, max=self.max)

        return loss, out_result

class ComputeLossBinOTA:
    # Compute losses
    def __init__(self, cfg, device='cuda:0'):
        super(ComputeLossBinOTA, self).__init__()
        
        # Set Config
        self.hyp = self.cfg.hyp
        self.nc = 80
        self.na = 3 # number of anchors
        self.stride = torch.tensor([8., 16., 32.])
        self.nl = 3 # number of detection layers (small, medium, large) =3
        self.ssi = 0
        self.gr = 1.0
        self.anchors = torch.tensor([
                        [[1.50000, 2.00000], [2.37500, 4.50000], [5.00000, 3.50000]],
                        [[2.25000, 4.68750], [4.75000, 3.43750], [4.50000, 9.12500]],
                        [[4.43750, 3.43750], [6.00000, 7.59375], [14.34375, 12.53125]]
                    ], device='cpu')

        # Define criteria
        # 1. cls loss
        with open('cfg/ava_categories_ratio.json', 'r') as fb:
            self.class_ratio = json.load(fb)
        self.register_buffer('class_weight', torch.zeros(80))
        for i in range(1, 81):
            self.class_weight[i - 1] = 1 - self.class_ratio[str(i)]
        self.BCEcls = nn.BCEWithLogitsLoss(pos_weight=self.class_weights.to(device), device='cuda:0')
        
        # 2. obj loss
        self.balance = [4.0, 1.0, 0.4]
        self.BCEobj = nn.BCEWithLogitsLoss()

    
    def forward(self, p_cls, p_bbox, t_cls, t_bbox):
        t_cls = convert_one_hot_to_batch_class(t_cls)
        t_bbox = extract_bounding_boxes(t_bbox)
        
        if isinstance(t_cls, np.ndarray):
            t_bbox = torch.Tensor(t_bbox)
            t_cls = torch.Tensor(t_cls)
            
        targets = torch.cat((t_bbox, t_cls[..., 1:]), dim=1)
        
        p = [torch.cat((bbox, cls), dim=4) for bbox, cls in zip(p_bbox, p_cls)]

        total_loss, loss_items = self.__call__(p, targets)
        
        return total_loss, loss_items

    def __call__(self, p, targets):  # predictions, targets, model   
        device = targets.device
        lcls, lbox, lobj = torch.zeros(1, device=device), torch.zeros(1, device=device), torch.zeros(1, device=device)
        bs, as_, gjs, gis, targets, anchors = self.build_targets(p, targets, img_shape=224)
        pre_gen_gains = [torch.tensor(pp.shape, device=device)[[3, 2, 3, 2]] for pp in p] 
    

        # Losses
        for i, pi in enumerate(p):  # layer index, layer predictions
            b, a, gj, gi = bs[i], as_[i], gjs[i], gis[i]  # image, anchor, gridy, gridx
            tobj = torch.zeros_like(pi[..., 0], device=device)  # target obj

            obj_idx = self.wh_bin_sigmoid.get_length()*2 + 2     # x,y, w-bce, h-bce     # xy_bin_sigmoid.get_length()*2

            n = b.shape[0]  # number of targets
            if n:
                ps = pi[b, a, gj, gi]  # prediction subset corresponding to targets

                # Regression
                grid = torch.stack([gi, gj], dim=1)
                selected_tbox = targets[i][:, 2:6] * pre_gen_gains[i]
                selected_tbox[:, :2] -= grid

                w_loss, pw = self.wh_bin_sigmoid.training_loss(ps[..., 2:(3+self.bin_count)], selected_tbox[..., 2] / anchors[i][..., 0])
                h_loss, ph = self.wh_bin_sigmoid.training_loss(ps[..., (3+self.bin_count):obj_idx], selected_tbox[..., 3] / anchors[i][..., 1])

                pw *= anchors[i][..., 0]
                ph *= anchors[i][..., 1]

                px = ps[:, 0].sigmoid() * 2. - 0.5
                py = ps[:, 1].sigmoid() * 2. - 0.5

                lbox += w_loss + h_loss # + x_loss + y_loss
                pbox = torch.cat((px.unsqueeze(1), py.unsqueeze(1), pw.unsqueeze(1), ph.unsqueeze(1)), 1).to(device)  # predicted box
                iou = bbox_iou(pbox.T, selected_tbox, x1y1x2y2=False, CIoU=True)  # iou(prediction, target)
                lbox += (1.0 - iou).mean()  # iou loss

                # Objectness
                tobj[b, a, gj, gi] = (1.0 - self.gr) + self.gr * iou.detach().clamp(0).type(tobj.dtype)  # iou ratio

                # Classification
                selected_tcls = targets[i][:, 1].long()
                if self.nc > 1:  # cls loss (only if multiple classes)
                    t = torch.full_like(ps[:, (1+obj_idx):], self.cn, device=device)  # targets
                    t[range(n), selected_tcls] = self.cp
                    lcls += self.BCEcls(ps[:, (1+obj_idx):], t)  # BCE

            obji = self.BCEobj(pi[..., obj_idx], tobj)
            lobj += obji * self.balance[i]  # obj loss

        lbox *= self.hyp['box']
        lobj *= self.hyp['obj']
        lcls *= self.hyp['cls']
        bs = tobj.shape[0]  # batch size

        loss = lbox + lobj + lcls
        return loss * bs, torch.cat((lbox, lobj, lcls, loss)).detach() # shape of (4, )

    def build_targets(self, p, targets, img_shape=224):
        
        #indices, anch = self.find_positive(p, targets)
        indices, anch = self.find_3_positive(p, targets)
        #indices, anch = self.find_4_positive(p, targets)
        #indices, anch = self.find_5_positive(p, targets)
        #indices, anch = self.find_9_positive(p, targets)

        matching_bs = [[] for pp in p]
        matching_as = [[] for pp in p]
        matching_gjs = [[] for pp in p]
        matching_gis = [[] for pp in p]
        matching_targets = [[] for pp in p]
        matching_anchs = [[] for pp in p]
        
        nl = len(p)    
    
        for batch_idx in range(p[0].shape[0]):
        
            b_idx = targets[:, 0]==batch_idx
            this_target = targets[b_idx]
            if this_target.shape[0] == 0:
                continue
                
            txywh = this_target[:, 2:6] * img_shape
            txyxy = xywh2xyxy(txywh)

            pxyxys = []
            p_cls = []
            p_obj = []
            from_which_layer = []
            all_b = []
            all_a = []
            all_gj = []
            all_gi = []
            all_anch = []
            
            for i, pi in enumerate(p):
                
                obj_idx = self.wh_bin_sigmoid.get_length()*2 + 2
                
                b, a, gj, gi = indices[i]
                idx = (b == batch_idx)
                b, a, gj, gi = b[idx], a[idx], gj[idx], gi[idx]                
                all_b.append(b)
                all_a.append(a)
                all_gj.append(gj)
                all_gi.append(gi)
                all_anch.append(anch[i][idx])
                from_which_layer.append(torch.ones(size=(len(b),)) * i)
                
                fg_pred = pi[b, a, gj, gi]                
                p_obj.append(fg_pred[:, obj_idx:(obj_idx+1)])
                p_cls.append(fg_pred[:, (obj_idx+1):])
                
                grid = torch.stack([gi, gj], dim=1)
                pxy = (fg_pred[:, :2].sigmoid() * 2. - 0.5 + grid) * self.stride[i] #/ 8.
                #pwh = (fg_pred[:, 2:4].sigmoid() * 2) ** 2 * anch[i][idx] * self.stride[i] #/ 8.
                pw = self.wh_bin_sigmoid.forward(fg_pred[..., 2:(3+self.bin_count)].sigmoid()) * anch[i][idx][:, 0] * self.stride[i]
                ph = self.wh_bin_sigmoid.forward(fg_pred[..., (3+self.bin_count):obj_idx].sigmoid()) * anch[i][idx][:, 1] * self.stride[i]
                
                pxywh = torch.cat([pxy, pw.unsqueeze(1), ph.unsqueeze(1)], dim=-1)
                pxyxy = xywh2xyxy(pxywh)
                pxyxys.append(pxyxy)
            
            pxyxys = torch.cat(pxyxys, dim=0)
            if pxyxys.shape[0] == 0:
                continue
            p_obj = torch.cat(p_obj, dim=0)
            p_cls = torch.cat(p_cls, dim=0)
            from_which_layer = torch.cat(from_which_layer, dim=0)
            all_b = torch.cat(all_b, dim=0)
            all_a = torch.cat(all_a, dim=0)
            all_gj = torch.cat(all_gj, dim=0)
            all_gi = torch.cat(all_gi, dim=0)
            all_anch = torch.cat(all_anch, dim=0)
        
            pair_wise_iou = box_iou(txyxy, pxyxys)

            pair_wise_iou_loss = -torch.log(pair_wise_iou + 1e-8)

            top_k, _ = torch.topk(pair_wise_iou, min(10, pair_wise_iou.shape[1]), dim=1)
            dynamic_ks = torch.clamp(top_k.sum(1).int(), min=1)

            gt_cls_per_image = (
                F.one_hot(this_target[:, 1].to(torch.int64), self.nc)
                .float()
                .unsqueeze(1)
                .repeat(1, pxyxys.shape[0], 1)
            )

            num_gt = this_target.shape[0]            
            cls_preds_ = (
                p_cls.float().unsqueeze(0).repeat(num_gt, 1, 1).sigmoid_()
                * p_obj.unsqueeze(0).repeat(num_gt, 1, 1).sigmoid_()
            )

            y = cls_preds_.sqrt_()
            pair_wise_cls_loss = F.binary_cross_entropy_with_logits(
               torch.log(y/(1-y)) , gt_cls_per_image, reduction="none"
            ).sum(-1)
            del cls_preds_
        
            cost = (
                pair_wise_cls_loss
                + 3.0 * pair_wise_iou_loss
            )

            matching_matrix = torch.zeros_like(cost)

            for gt_idx in range(num_gt):
                _, pos_idx = torch.topk(
                    cost[gt_idx], k=dynamic_ks[gt_idx].item(), largest=False
                )
                matching_matrix[gt_idx][pos_idx] = 1.0

            del top_k, dynamic_ks
            anchor_matching_gt = matching_matrix.sum(0)
            if (anchor_matching_gt > 1).sum() > 0:
                _, cost_argmin = torch.min(cost[:, anchor_matching_gt > 1], dim=0)
                matching_matrix[:, anchor_matching_gt > 1] *= 0.0
                matching_matrix[cost_argmin, anchor_matching_gt > 1] = 1.0
            fg_mask_inboxes = matching_matrix.sum(0) > 0.0
            matched_gt_inds = matching_matrix[:, fg_mask_inboxes].argmax(0)
        
            from_which_layer = from_which_layer[fg_mask_inboxes]
            all_b = all_b[fg_mask_inboxes]
            all_a = all_a[fg_mask_inboxes]
            all_gj = all_gj[fg_mask_inboxes]
            all_gi = all_gi[fg_mask_inboxes]
            all_anch = all_anch[fg_mask_inboxes]
        
            this_target = this_target[matched_gt_inds]
        
            for i in range(nl):
                layer_idx = from_which_layer == i
                matching_bs[i].append(all_b[layer_idx])
                matching_as[i].append(all_a[layer_idx])
                matching_gjs[i].append(all_gj[layer_idx])
                matching_gis[i].append(all_gi[layer_idx])
                matching_targets[i].append(this_target[layer_idx])
                matching_anchs[i].append(all_anch[layer_idx])

        for i in range(nl):
            if matching_targets[i] != []:
                matching_bs[i] = torch.cat(matching_bs[i], dim=0)
                matching_as[i] = torch.cat(matching_as[i], dim=0)
                matching_gjs[i] = torch.cat(matching_gjs[i], dim=0)
                matching_gis[i] = torch.cat(matching_gis[i], dim=0)
                matching_targets[i] = torch.cat(matching_targets[i], dim=0)
                matching_anchs[i] = torch.cat(matching_anchs[i], dim=0)
            else:
                matching_bs[i] = torch.tensor([], device='cuda:0', dtype=torch.int64)
                matching_as[i] = torch.tensor([], device='cuda:0', dtype=torch.int64)
                matching_gjs[i] = torch.tensor([], device='cuda:0', dtype=torch.int64)
                matching_gis[i] = torch.tensor([], device='cuda:0', dtype=torch.int64)
                matching_targets[i] = torch.tensor([], device='cuda:0', dtype=torch.int64)
                matching_anchs[i] = torch.tensor([], device='cuda:0', dtype=torch.int64)

        return matching_bs, matching_as, matching_gjs, matching_gis, matching_targets, matching_anchs       

    def find_3_positive(self, p, targets):
        # Build targets for compute_loss(), input targets(image,class,x,y,w,h)
        na, nt = self.na, targets.shape[0]  # number of anchors, targets
        indices, anch = [], []
        gain = torch.ones(7, device=targets.device).long()  # normalized to gridspace gain
        ai = torch.arange(na, device=targets.device).float().view(na, 1).repeat(1, nt)  # same as .repeat_interleave(nt)
        targets = torch.cat((targets.repeat(na, 1, 1), ai[:, :, None]), 2)  # append anchor indices

        g = 0.5  # bias
        off = torch.tensor([[0, 0],
                            [1, 0], [0, 1], [-1, 0], [0, -1],  # j,k,l,m
                            # [1, 1], [1, -1], [-1, 1], [-1, -1],  # jk,jm,lk,lm
                            ], device=targets.device).float() * g  # offsets

        for i in range(self.nl):
            anchors = self.anchors[i]
            gain[2:6] = torch.tensor(p[i].shape)[[3, 2, 3, 2]]  # xyxy gain

            # Match targets to anchors
            t = targets * gain
            if nt:
                # Matches
                r = t[:, :, 4:6] / anchors[:, None]  # wh ratio
                j = torch.max(r, 1. / r).max(2)[0] < self.hyp['anchor_t']  # compare
                # j = wh_iou(anchors, t[:, 4:6]) > model.hyp['iou_t']  # iou(3,n)=wh_iou(anchors(3,2), gwh(n,2))
                t = t[j]  # filter

                # Offsets
                gxy = t[:, 2:4]  # grid xy
                gxi = gain[[2, 3]] - gxy  # inverse
                j, k = ((gxy % 1. < g) & (gxy > 1.)).T
                l, m = ((gxi % 1. < g) & (gxi > 1.)).T
                j = torch.stack((torch.ones_like(j), j, k, l, m))
                t = t.repeat((5, 1, 1))[j]
                offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]
            else:
                t = targets[0]
                offsets = 0

            # Define
            b, c = t[:, :2].long().T  # image, class
            gxy = t[:, 2:4]  # grid xy
            gwh = t[:, 4:6]  # grid wh
            gij = (gxy - offsets).long()
            gi, gj = gij.T  # grid xy indices

            # Append
            a = t[:, 6].long()  # anchor indices
            indices.append((b, a, gj.clamp_(0, gain[3] - 1), gi.clamp_(0, gain[2] - 1)))  # image, anchor, grid indices
            anch.append(anchors[a])  # anchors

        return indices, anch



if __name__ == '__main__':
    pred_boxes = torch.randn((2352,4), dtype=torch.float32)
    array = np.zeros((1, 50, 80), dtype=np.float32)
    for i in range(6):
        one_hot_encoding = np.zeros(80, dtype=np.float32)
        one_hot_encoding[i] = 1.0
        array[:, i, :] = one_hot_encoding
    
    target = {
    'cls': torch.Tensor(array),
    'boxes': torch.Tensor(np.array([[
        [       0.18,       0.481,       0.206,        0.66],
        [      0.296,      0.2645,        0.14,       0.465],
        [     0.4065,      0.5425,       0.149,       0.697],
        [      0.579,      0.4425,       0.148,       0.675],
        [     0.7155,       0.482,       0.179,       0.672],
        [    0.90027,     0.60877,     0.19054,     0.77354],
        [          0,           0,           0,           0],
        [          0,           0,           0,           0],
        [          0,           0,           0,           0],
        [          0,           0,           0,           0],
        [          0,           0,           0,           0],
        [          0,           0,           0,           0],
        [          0,           0,           0,           0],
        [          0,           0,           0,           0],
        [          0,           0,           0,           0],
        [          0,           0,           0,           0],
        [          0,           0,           0,           0],
        [          0,           0,           0,           0],
        [          0,           0,           0,           0],
        [          0,           0,           0,           0],
        [          0,           0,           0,           0],
        [          0,           0,           0,           0],
        [          0,           0,           0,           0],
        [          0,           0,           0,           0],
        [          0,           0,           0,           0],
        [          0,           0,           0,           0],
        [          0,           0,           0,           0],
        [          0,           0,           0,           0],
        [          0,           0,           0,           0],
        [          0,           0,           0,           0],
        [          0,           0,           0,           0],
        [          0,           0,           0,           0],
        [          0,           0,           0,           0],
        [          0,           0,           0,           0],
        [          0,           0,           0,           0],
        [          0,           0,           0,           0],
        [          0,           0,           0,           0],
        [          0,           0,           0,           0],
        [          0,           0,           0,           0],
        [          0,           0,           0,           0],
        [          0,           0,           0,           0],
        [          0,           0,           0,           0],
        [          0,           0,           0,           0],
        [          0,           0,           0,           0],
        [          0,           0,           0,           0],
        [          0,           0,           0,           0],
        [          0,           0,           0,           0],
        [          0,           0,           0,           0],
        [          0,           0,           0,           0],
        [          0,           0,           0,           0]]], dtype=np.float32))}
    anchors = [1.28967, 4.15014, 2.12714, 5.09344, 3.27212, 5.87423]
    num_anchors = 3
    num_classes = 80
    object_scale = 5
    nH = 28
    nW = 28
    noobject_scale = 1
    sil_thresh = 0.6
    
    ret = build_targets_Ava(pred_boxes, target, anchors, num_anchors, num_classes, nH, nW, noobject_scale, object_scale, sil_thresh)
    print(ret)