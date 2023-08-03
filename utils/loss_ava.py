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
    def __init__(self, cfg, device='cuda:0'):
        super(ComputeLoss, self).__init__()
        
        # Set Config
        self.hyp = cfg.hyp
        self.nc = None
        self.na = len(cfg.MODEL.ANCHORS[0]) // 2 # number of anchors
        self.nl = len(cfg.MODEL.ANCHORS) # number of layers
        self.ssi = 0
        self.gr = 1.0
        self.cp, self.cn = 0.95, 0.05 # Smooth BCE
        self.anchors = torch.Tensor(cfg.MODEL.ANCHORS).view(self.nl, self.na, 2).to(device)

        # Define criteria
        # 1. cls loss
        with open('cfg/ava_categories_ratio.json', 'r') as fb:
            self.class_ratio = json.load(fb)
        self.class_weight = torch.zeros(80)
        for i in range(1, 81):
            self.class_weight[i - 1] = 1 - self.class_ratio[str(i)]
        self.BCEcls_ava = nn.BCEWithLogitsLoss(pos_weight=self.class_weight.to(device))
        self.BCEcls_df2 = nn.BCEWithLogitsLoss()
        
        # 2. obj loss
        self.balance = [4.0, 1.0, 0.4]
        self.BCEobj = nn.BCEWithLogitsLoss()

    
    def forward_ava(self, p_cls, p_bbox, t_cls, t_bbox):
        self.nc = 80
        self.BCEcls = self.BCEcls_ava
        t_cls = convert_one_hot_to_batch_class(t_cls)
        t_bbox = extract_bounding_boxes(t_bbox)
        
        if isinstance(t_cls, np.ndarray):
            t_bbox = torch.Tensor(t_bbox)
            t_cls = torch.Tensor(t_cls)
            
        targets = torch.cat((t_cls, t_bbox[..., 1:]), dim=1).to('cuda:0')
        
        p = [torch.cat((bbox, cls), dim=4) for bbox, cls in zip(p_bbox, p_cls)]

        total_loss, loss_items = self.__call__(p, targets)
        
        return total_loss, loss_items
    
    def forward_df2(self, p_cls, p_bbox, targets):
        self.nc = 13
        self.BCEcls = self.BCEcls_df2
        p = [torch.cat((bbox, cls), dim=4) for bbox, cls in zip(p_bbox, p_cls)]
        targets = targets.to('cuda:0')
        total_loss, loss_items = self.__call__(p, targets)
        
        return total_loss, loss_items


    def __call__(self, p, targets):  # predictions, targets, model
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
                    lcls += self.BCEcls(ps[:, 5:], t)  # BCE

                # Append targets to text file
                # with open('targets.txt', 'a') as file:
                #     [file.write('%11.5g ' * 4 % tuple(x) + '\n') for x in torch.cat((txy[i], twh[i]), 1)]

            obji = self.BCEobj(pi[..., 4], tobj)
            lobj += obji * self.balance[i]  # obj loss

        lbox *= self.hyp['box']
        lobj *= self.hyp['obj']
        lcls *= self.hyp['cls']
        bs = tobj.shape[0]  # batch size

        loss = lbox + lobj + lcls
        return loss * bs, torch.cat((lbox, lobj, lcls, loss))

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
            tbox.append(torch.cat((gxy - gij, gwh), 1))  # box
            anch.append(anchors[a])  # anchors
            tcls.append(c)  # class

        return tcls, tbox, indices, anch
