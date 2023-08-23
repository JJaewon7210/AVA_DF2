import time
import json
import torch
import torch.nn.functional as F

import math
import torch.nn as nn
from torch.autograd import Variable
from builtins import range as xrange
from utils.general import bbox_iou, xywh2xyxy, box_iou
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

def convert2cpu(gpu_matrix):
    return torch.FloatTensor(gpu_matrix.size()).copy_(gpu_matrix)

class FocalLoss(nn.Module):
    # Wraps focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super(FocalLoss, self).__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        # p_t = torch.exp(-loss)
        # loss *= self.alpha * (1.000001 - p_t) ** self.gamma  # non-zero power for gradient stability

        # TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
        pred_prob = torch.sigmoid(pred)  # prob from logits
        p_t = true * pred_prob + (1 - true) * (1 - pred_prob)
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss

class WeightedMultiTaskLoss(nn.Module):
    def __init__(self, num_tasks):
        super().__init__()
        self.eta = nn.Parameter(torch.zeros(num_tasks, device=torch.device('cuda:0')))

    def forward(self, losses):
        '''
        Parameters:
            - losses (list): List of losses where each type of loss is defined as a Torch.Tensor.
            For instance, [loss01, loss02, loss03].
        Returns:
            - Tuple of two items:
                1. The original losses (torch.Tensor): The input losses without weighting.
                2. The combined loss (torch.Tensor): The total loss that combines individual losses 
                   using learnable loss weights.
        '''
        losses = [loss.to(torch.device('cuda:0')) for loss in losses]
        
        weighted_losses = torch.stack(losses).squeeze() * torch.exp(-self.eta) + self.eta
        combined_loss = weighted_losses.sum()
        return combined_loss

class ComputeLoss:
    # Compute losses
    def __init__(self, detector_head, hyp, device='cuda:0'):
        super(ComputeLoss, self).__init__()
        
        # Set Config
        det = detector_head
        for k in 'na', 'nl', 'anchors':
            setattr(self, k, getattr(det, k))
            
        self.ssi = 0 
        self.gr = 0 # iou loss ratio (obj_loss = 1.0 or iou)

        # Define criteria
        BCEcls_ava = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([hyp['cls_pw']], device=device))
        BCEcls_df2 = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([hyp['cls_pw']], device=device))
        BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([hyp['obj_pw']], device=device))
        self.cp, self.cn = 1.0, 0.0 # Smooth BCE
        
        # Focal loss
        g = hyp['fl_gamma']  # focal loss gamma
        if g > 0:
            BCEcls_ava, BCEcls_df2, BCEobj = FocalLoss(BCEcls_ava, g), FocalLoss(BCEcls_df2, g), FocalLoss(BCEobj, g)
        
        self.BCEcls_ava, self.BCEcls_df2, self.BCEobj, self.hyp = BCEcls_ava, BCEcls_df2, BCEobj, hyp
        self.anchors = self.anchors.to(device)
    
    def forward_ava(self, p_cls, p_bbox, t_cls, t_bbox):
        self.nc = 80
        t_cls = convert_one_hot_to_batch_class(t_cls)
        t_bbox = extract_bounding_boxes(t_bbox)
        
        if isinstance(t_cls, np.ndarray):
            t_bbox = torch.Tensor(t_bbox)
            t_cls = torch.Tensor(t_cls)
            
        targets = torch.cat((t_cls, t_bbox[..., 1:]), dim=1).to('cuda:0')
        
        p = [torch.cat((bbox, cls), dim=4) for bbox, cls in zip(p_bbox, p_cls)]

        total_loss, loss_items = self.__call__(p, targets, self.BCEcls_ava)
        
        return total_loss, loss_items
    
    def forward_df2(self, p_cls, p_bbox, targets):
        self.nc = 13
        p = [torch.cat((bbox, cls), dim=4) for bbox, cls in zip(p_bbox, p_cls)]
        targets = targets.to('cuda:0')
        total_loss, loss_items = self.__call__(p, targets, self.BCEcls_df2)
        
        return total_loss, loss_items


    def __call__(self, p, targets, BCEcls):  # predictions, targets, model
        device = targets.device
        lcls, lbox, lobj = torch.zeros(1, device=device), torch.zeros(1, device=device), torch.zeros(1, device=device)
        tcls, tbox, indices, anchors = self.build_targets(p, targets)  # targets

        # Losses
        for i, pi in enumerate(p):  # layer index, layer predictions
            b, a, gj, gi = indices[i]  # image, anchor, gridy, gridx
            tobj = torch.zeros_like(pi[..., 0], device=device)  # target obj

            n = b.shape[0]  # number of targets
            if n:
                ps = pi[b, a, gj, gi]  # prediction subset corresponding to targets

                # Regression
                pxy = ps[:, :2].sigmoid() * 2. - 0.5
                pwh = (ps[:, 2:4].sigmoid() * 2) ** 2 * anchors[i]
                pbox = torch.cat((pxy, pwh), 1)  # predicted box
                iou = bbox_iou(pbox.T, tbox[i], x1y1x2y2=False, CIoU=True)  # iou(prediction, target)
                lbox += (1.0 - iou).mean()  # iou loss

                # Objectness
                tobj[b, a, gj, gi] = (1.0 - self.gr) + self.gr * iou.detach().clamp(0).type(tobj.dtype)  # iou ratio

                # Classification
                if self.nc > 1:  # cls loss (only if multiple classes)
                    t = torch.full_like(ps[:, 5:], self.cn, device=device)  # targets
                    t[range(n), tcls[i]] = self.cp
                    #t[t==self.cp] = iou.detach().clamp(0).type(t.dtype)
                    lcls += BCEcls(ps[:, 5:], t)  # BCE

                # Append targets to text file
                # with open('targets.txt', 'a') as file:
                #     [file.write('%11.5g ' * 4 % tuple(x) + '\n') for x in torch.cat((txy[i], twh[i]), 1)]

            obji = self.BCEobj(pi[..., 4], tobj)
            lobj += obji # obj loss

        lbox *= self.hyp['box']
        lobj *= self.hyp['obj']
        lcls *= self.hyp['cls']
        bs = tobj.shape[0]  # batch size

        loss = lbox + lobj + lcls
        return loss * bs, torch.cat((lbox, lobj, lcls, loss))
    
    def generate_grid_within_bbox(self, bbox):
        xc, yc, width, height = bbox
        xmin = int(xc - width / 2)
        xmax = int(xc + width / 2)
        ymin = int(yc - height / 2)
        ymax = int(yc + height / 2)
        
        x_points = torch.arange(xmin, xmax + 1, dtype=torch.int32)
        y_points = torch.arange(ymin, ymax + 1, dtype=torch.int32)
        
        grid = torch.meshgrid(x_points, y_points, indexing='ij')  # Pass indexing='ij'
        grid = torch.stack(grid, dim=-1).view(-1, 2)
        
        # Subtract center coordinates from each grid point
        center_coords = torch.tensor([int(xc), int(yc)])
        grid = grid - center_coords
        
        return grid.to(device = 'cuda:0')

    def build_targets(self, p, targets):
        # Build targets for compute_loss(), input targets(image,class,x,y,w,h)
        na, nt = self.na, targets.shape[0]  # number of anchors, targets
        tcls, tbox, indices, anch = [], [], [], []
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
            t = targets * gain # na, nt, 7
            if nt:
                t = t.view(na*nt, 7)  # Reshape t to (na*nt, 7)
                
                t_after = []
                offsets_after = []
                
                for ti in range(na*nt):
                    t_ = t[ti, ... ]
                    bbox = t[ti, 2:6]
                    gxy = t[ti, 2:4]
                    off = self.generate_grid_within_bbox(bbox)
                    
                    j = torch.ones((len(off))).bool()
                    t_new = t_.repeat(len(off), 1, 1)[j]
                    offsets_new = (torch.zeros_like(gxy)[None] + off[:, None])[j]
                    
                    t_after.append(t_new)
                    offsets_after.append(offsets_new)
                    
                t = torch.cat(t_after).squeeze()
                offsets = torch.cat(offsets_after).squeeze()
                
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
            tbox.append(torch.cat((gxy - gij, gwh), 1))  # box
            anch.append(anchors[a])  # anchors
            tcls.append(c)  # class

        return tcls, tbox, indices, anch
