import sys
import os
# Append the mother directory path to sys.path
sys.path.append('C:/CNN/AVA_DF2')

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.general import bbox_iou, bbox_alpha_iou, box_iou, box_giou, box_diou, box_ciou, xywh2xyxy, box_iou_only_box1

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
        
        weighted_losses = torch.stack(losses) * torch.exp(-self.eta) + self.eta
        combined_loss = weighted_losses.sum()
        return combined_loss

    

class CustomLoss_:
    def __init__(self):
        super(CustomLoss_, self).__init__()
        
        self.nc = 13 # number of cloth classes (13)
        self.na = 3 # number of anchors
        self.stride = torch.tensor([8., 16., 32.])
        self.nl = 3 # number of detection layers (small, medium, large) =3
        self.ssi = 0
        self.gr = 1.0
        
        # Class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
        self.cp, self.cn = 1.0, 0.0 # smooth_BCE(eps=h.get('label_smoothing', 0.0))  # positive, negative BCE targets
        
        self.autobalance = False
        self.balance = [4.0, 1.0, 0.4]
        self.anchors = torch.tensor([
                        [[1.50000, 2.00000], [2.37500, 4.50000], [5.00000, 3.50000]],
                        [[2.25000, 4.68750], [4.75000, 3.43750], [4.50000, 9.12500]],
                        [[4.43750, 3.43750], [6.00000, 7.59375], [14.34375, 12.53125]]
                    ], device='cpu')
        self.hyp = {'lr0': 0.01, 'lrf': 0.1, 
                    'momentum': 0.937, 'weight_decay': 0.0005, 
                    'warmup_epochs': 3.0, 'warmup_momentum': 0.8, 'warmup_bias_lr': 0.1, 
                    'box': 0.05, 
                    'cls': 0.04875, 'cls_pw': 1.0, 
                    'obj': 0.08575, 'obj_pw': 1.0, 
                    'iou_t': 0.2, 'anchor_t': 40.0}
        
        self.BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([self.hyp['cls_pw']], device='cpu'))
    
    def loss_class_cloth(self, prediction, targets, pseudo_label):
        """
        This function calculates the binary cross-entropy loss for object classification in the context of object detection.

        Parameters:
        - prediction: A list of tensors. 
            Each tensor represents the prediction for a different scale, 
            and has shape [batch size, number of anchors, grid size, grid size, number of classes].
        - target: A list of tensors. 
            Each tensor contains the targets for a different scale, 
            with the same dimensions as the corresponding prediction tensor.
        - pseudo_label: A list of tensors.
            Each tensor represents the bbox localization for a different scale,
            and has shape [batch size, number of anchors, grid size, grid size, 4+1]

        Returns:
        - loss_cls: A scalar tensor representing the binary cross-entropy loss for the object classification task.

        Note:
        This function expects the target labels to be in one-hot encoded format.
        """
        device = targets.device
        loss_cls = torch.zeros(1, device=device)
        bs, as_, gjs, gis, targets, anchors = self.build_targets(pseudo_label, targets, img_shape=224)
    
        # Losses
        for i, pi in enumerate(prediction):  # layer index, layer predictions
            b, a, gj, gi = bs[i], as_[i], gjs[i], gis[i]  # image, anchor, gridy, gridx

            n = b.shape[0]  # number of targets
            if n:
                ps = pi[b, a, gj, gi]  # prediction subset corresponding to targets

                # Classification
                selected_tcls = targets[i][:, 1].long()
                if self.nc > 1:  # cls loss (only if multiple classes)
                    t = torch.full_like(ps, self.cn, device=device)  # targets
                    t[range(n), selected_tcls] = self.cp
                    loss_cls += self.BCEcls(ps, t)  # BCE
                    
        return loss_cls
    
       
    def loss_feature_distance(self, prediction, target):
        """
        This function calculates the Mean Squared Error (MSE) loss between predicted and target features for BBox localization.
        
        Parameters:
        - prediction: A dictionary containing the predicted features for small, medium and large BBoxes. 
                    Each entry is a tensor of shape [Batch, 3, *, *, (4+1)], 
                    where '*' could be 7, 14, or 28 depending on the feature map size.
        - target: A dictionary containing the ground truth features for small, medium and large BBoxes. 
                    Each entry is a tensor of shape [Batch, 3, *, *, (4+1)], 
                    where '*' could be 7, 14, or 28 depending on the feature map size.

        Returns:
        - total_loss: A scalar tensor representing the total MSE loss across all feature map sizes.

        Note: The (4+1) in the tensor shape refers to the 4 coordinates (x,y,h,w) and 1 confidence score for each anchor box.
        """
        
        # Extract features from prediction
        p_feature_s = prediction['feature_s']
        p_feature_m = prediction['feature_m']
        p_feature_l = prediction['feature_l']
        
        # Extract features from target
        gt_feature_s = target['feature_s']
        gt_feature_m = target['feature_m']
        gt_feature_l = target['feature_l']

        # Assert that prediction and target tensors have the same shape
        assert p_feature_s.shape == gt_feature_s.shape, "Mismatch in shape for feature_s"
        assert p_feature_m.shape == gt_feature_m.shape, "Mismatch in shape for feature_m"
        assert p_feature_l.shape == gt_feature_l.shape, "Mismatch in shape for feature_l"

        # Calculate mean square error for each feature map
        loss_s = (p_feature_s - gt_feature_s).pow(2).mean()
        loss_m = (p_feature_m - gt_feature_m).pow(2).mean()
        loss_l = (p_feature_l - gt_feature_l).pow(2).mean()

        # Total loss is the sum of each feature map's loss
        total_loss = loss_s + loss_m + loss_l
        
        return total_loss


    def build_targets(self, p, targets, img_shape=224):
        '''Builds the targets for the YOLO-like detector.

        Parameters:
            - `p (List[torch.Tensor])`: List of model predictions 
                - Example shape: [[Batch, num_anchors, 28, 28, (4+1+13+80)], [Batch, num_anchors, 14, 14, 98], [Batch, num_anchors, 7, 7, 98]]
                - p is a list containing three tensors, each representing model predictions for a specific scale grid. 
                - The first tensor has shape [Batch, num_anchors, 28, 28, (4+1+13+80)], where:
                
                    - Batch: The batch size.
                    - num_anchors: The number of anchor boxes per grid cell.
                    - 28, 28: The grid size of the first scale.
                    - (4+1+13+80): The number of prediction channels for each anchor, including:
                        - 4 channels for the bounding box coordinates (x, y, width, height).
                        - 1 channel for the confidence score (objectness probability).
                        - 13 channels for class probabilities (corresponding to the number of cloth classes in the deepfashion2 dataset).
                        - 80 channels for class probabilities (corresponding to the number of action classes in the AVA2.2 dataset).

                The second and third tensors have similar shapes but with different grid sizes.

            - `targets (torch.Tensor)`: Ground truth target values for the batch of shape 
                - Example shape: [num_target, (1+4+13+80)].
                - targets is a tensor representing the ground truth target values for the batch.
                
                    - num_target: total target size in single batch.
                    - 1+4+13+80: The total number of target channels per target, including:
                        - 1 channel for the image index (batch index)
                        - 4 channels for the bounding box coordinates (x, y, width, height).
                        - 13 channels for cloth class probabilities.
                        - 80 channels for action class probabilities.
        
            - `img_shape (int, optional)`: The shape of the input image (square image). Defaults to 224.

        Returns:
            tuple: Tuple containing lists of matching indices and corresponding targets for each YOLO layer.
            
        Output Details:
            - matching_bs (List of torch.Tensor): A list of tensors of shape [num_matched_boxes], where num_matched_boxes
                                                is the number of bounding boxes successfully matched to ground truth
                                                targets on each YOLO layer. Each element represents the batch index (image index)
                                                to which the matched box belongs.

            - matching_as (List of torch.Tensor): A list of tensors of shape [num_matched_boxes], where num_matched_boxes
                                                is the number of bounding boxes successfully matched to ground truth
                                                targets on each YOLO layer. Each element represents the anchor index of the matched
                                                box on the corresponding YOLO layer.

            - matching_gjs (List of torch.Tensor): A list of tensors of shape [num_matched_boxes], where num_matched_boxes
                                                is the number of bounding boxes successfully matched to ground truth
                                                targets on each YOLO layer. Each element represents the grid cell y-coordinate index
                                                of the matched box on the corresponding YOLO layer.

            - matching_gis (List of torch.Tensor): A list of tensors of shape [num_matched_boxes], where num_matched_boxes
                                                is the number of bounding boxes successfully matched to ground truth
                                                targets on each YOLO layer. Each element represents the grid cell x-coordinate index
                                                of the matched box on the corresponding YOLO layer.

            - matching_targets (List of torch.Tensor): A list of tensors of shape [num_matched_boxes, (1+4+13+80)], where num_matched_boxes
                                                    is the number of bounding boxes successfully matched to ground truth
                                                    targets on each YOLO layer, and (1+4+13+80) is the total number of target channels
                                                    per target, as specified in the function's annotation. Each element represents
                                                    the ground truth target corresponding to the matched box.

            - matching_anchs (List of torch.Tensor): A list of tensors of shape [num_matched_boxes], where num_matched_boxes
                                                    is the number of bounding boxes successfully matched to ground truth
                                                    targets on each YOLO layer. Each element represents the anchor size (e.g., width and height)
                                                    of the matched box on the corresponding YOLO layer.

        Note: This function is specific to a YOLO-like detector and is designed to work with a list of predicted feature maps (`p`)
        from different YOLO layers. It performs target matching and generates lists of matching indices and corresponding targets
        for each YOLO layer. The outputs can be used to calculate the loss during training to update the detector's parameters.
        '''
        assert all(len(t.shape) == 5 for t in p), "Mismatch in the number of dimensions"
        assert len(set(t.shape[1] for t in p)) == 1, "Mismatch in the number of anchors between predictions"
        assert len(set(t.shape[4] for t in p)) == 1, "Mismatch in the number of channels between predictions"
        assert targets.shape[1] == p[0].shape[1] * p[0].shape[4], "Mismatch in target's second dimension size"
        
        targets_6 = torch.ones((targets.shape[0], 6), device=targets.device, dtype=torch.float32)
        targets_6[:,0] = targets[:,0] # image index (batch index)
        targets_6[:,1] = targets[:,5] # class num (not necessary)
        targets_6[:,2:6] = targets[:,1:5] # xywh
        indices, anch = self.find_3_positive(p, targets_6)

        device = torch.device(targets.device)
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
                
            # txywh = this_target[:, 1:5] * imgs[batch_idx].shape[1]
            txywh = this_target[:, 1:5] * img_shape
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
                
                b, a, gj, gi = indices[i]
                idx = (b == batch_idx)
                b, a, gj, gi = b[idx], a[idx], gj[idx], gi[idx]                
                all_b.append(b)
                all_a.append(a)
                all_gj.append(gj)
                all_gi.append(gi)
                all_anch.append(anch[i][idx])
                from_which_layer.append((torch.ones(size=(len(b),)) * i).to(device))
                
                fg_pred = pi[b, a, gj, gi]                
                p_obj.append(fg_pred[:, 4:5]) # 4+1+13+80 channel = 4 (xywh) + 1 (confidence score) + 13 (number of cloth classes) 
                # p_cls.append(fg_pred[:, 5:5+self.nc])
                
                grid = torch.stack([gi, gj], dim=1)
                pxy = (fg_pred[:, :2].sigmoid() * 2. - 0.5 + grid) * self.stride[i] #/ 8.
                pwh = (fg_pred[:, 2:4].sigmoid() * 2) ** 2 * anch[i][idx] * self.stride[i] #/ 8.
                pxywh = torch.cat([pxy, pwh], dim=-1)
                pxyxy = xywh2xyxy(pxywh)
                pxyxys.append(pxyxy)
            
            pxyxys = torch.cat(pxyxys, dim=0)
            if pxyxys.shape[0] == 0:
                continue
            p_obj = torch.cat(p_obj, dim=0)
            # p_cls = torch.cat(p_cls, dim=0)
            from_which_layer = torch.cat(from_which_layer, dim=0)
            all_b = torch.cat(all_b, dim=0)
            all_a = torch.cat(all_a, dim=0)
            all_gj = torch.cat(all_gj, dim=0)
            all_gi = torch.cat(all_gi, dim=0)
            all_anch = torch.cat(all_anch, dim=0)
        
            pair_wise_iou = box_iou_only_box1(txyxy, pxyxys, standard='box1') # txyxy [2, 4] x pxyxys [32, 4] = pair_wise_iou [2, 32]

            pair_wise_iou_loss = -torch.log(pair_wise_iou + 1e-8)

            # top_k, _ = torch.topk(pair_wise_iou, min(10, pair_wise_iou.shape[1]), dim=1) # [2, 32] -> [2, 10]
            # dynamic_ks = torch.clamp(top_k.sum(1).int(), min=1) # [2, 10] -> [2]
            top_k, _ = torch.topk(pair_wise_iou, min(20, pair_wise_iou.shape[1]), dim=1) # [2, 32] -> [2, 15]
            dynamic_ks = torch.clamp(top_k.sum(1).int(), min=10) # [2, 10] -> [2]

            # gt_cls_per_image = (
            #     F.one_hot(this_target[:, 1].to(torch.int64), self.nc)
            #     .float()
            #     .unsqueeze(1)
            #     .repeat(1, pxyxys.shape[0], 1)
            # )
            # gt_cls_per_image = (
            #     this_target[:, 5:5+self.nc].to(torch.int64)
            #     .float()
            #     .unsqueeze(1)
            #     .repeat(1, pxyxys.shape[0], 1)
            # )

            num_gt = this_target.shape[0]
            # cls_preds_ = (
            #     p_cls.float().unsqueeze(0).repeat(num_gt, 1, 1).sigmoid_()
            #     * p_obj.unsqueeze(0).repeat(num_gt, 1, 1).sigmoid_()
            # )

            # y = cls_preds_.sqrt_()
            # pair_wise_cls_loss = F.binary_cross_entropy_with_logits(
            #    torch.log(y/(1-y)) , gt_cls_per_image, reduction="none"
            # ).sum(-1)
            # del cls_preds_
        
            cost = (
                # pair_wise_cls_loss +
                3.0 * pair_wise_iou_loss
            )

            matching_matrix = torch.zeros_like(cost, device=device) # [2, 32]

            for gt_idx in range(num_gt):
                _, pos_idx = torch.topk(
                    cost[gt_idx], k=dynamic_ks[gt_idx].item(), largest=False
                )
                matching_matrix[gt_idx][pos_idx] = 1.0

            del top_k, dynamic_ks
            # anchor_matching_gt = matching_matrix.sum(0)
            # if (anchor_matching_gt > 1).sum() > 0:
            #     _, cost_argmin = torch.min(cost[:, anchor_matching_gt > 1], dim=0)
            #     matching_matrix[:, anchor_matching_gt > 1] *= 0.0
            #     matching_matrix[cost_argmin, anchor_matching_gt > 1] = 1.0
            fg_mask_inboxes = (matching_matrix.sum(0) > 0.0).to(device)
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


    def find_3_positive(self, p, targets_6):
        '''
        Find positive target anchor indices and corresponding anchor boxes for computing the loss during training.

        Parameters:
            - `p (List[torch.Tensor])`: List of model predictions 
                - Example shape: [[Batch, num_anchors, 28, 28, (4+1+13+80)], [Batch, num_anchors, 14, 14, 98], [Batch, num_anchors, 7, 7, 98]]
                - p is a list containing three tensors, each representing model predictions for a specific scale grid. 
                The first tensor has shape [Batch, num_anchors, 28, 28, (4+1+13+80)], where:
                    - Batch: The batch size.
                    - num_anchors: The number of anchor boxes per grid cell.
                    - 28, 28: The grid size of the first scale.
                    - (4+1+13+80): The number of prediction channels for each anchor, including:
                        - 4 channels for the bounding box coordinates (x, y, width, height).
                        - 1 channel for the confidence score (objectness probability).
                        - 13 channels for class probabilities (corresponding to the number of cloth classes in the deepfashion2 dataset).
                        - 80 channels for class probabilities (corresponding to the number of action classes in the AVA2.2 dataset).

                The second and third tensors have similar shapes but with different grid sizes.

            - `targets_6 (torch.Tensor)`: Ground truth target values for the batch of shape 
                - Example shape: [num_target, 1+1+4].
                - targets_6 is a tensor representing the ground truth target values for the batch.
                - num_target: total target size in single batch.
                - 1+1+4: The total number of target channels per target, including:
                    - 1 channel for the image index (batch index)
                    - 1 channel for the class index (integer value representing the target class). (not necessary)
                    - 4 channels for the bounding box coordinates (x, y, width, height).

        Returns:
            - Tuple[List[Tuple[torch.Tensor]], List[torch.Tensor]]:
                - A tuple containing two lists:
                    - `Indices` List of tuples containing positive anchor indices and grid indices for each image.
                        - Each tuple contains four tensors:
                        - torch.Tensor with shape [N], representing the image indices (batch indices).
                        - torch.Tensor with shape [N], representing the positive anchor indices for each image.
                        - torch.Tensor with shape [N], representing the grid indices (y-coordinate) for each positive anchor.
                        - torch.Tensor with shape [N], representing the grid indices (x-coordinate) for each positive anchor.
                    - `anch` List of corresponding anchor boxes for each positive anchor index.
                        - Each tensor has shape [N, 2, 3], representing the 3D anchor boxes (each box with 2 coordinates and 3 dimensions).
        '''
        na, nt = self.na, targets_6.shape[0]
        indices, anch = [], []
        gain = torch.ones(7, device=targets_6.device).long()
        ai = torch.arange(na, device=targets_6.device).float().view(na, 1).repeat(1, nt)
        targets_6 = torch.cat((targets_6.repeat(na, 1, 1), ai[:, :, None]), 2)  # append anchor indices [33, 6] -> [3, 33, 7]

        g = 0.5
        off = torch.tensor([[0, 0],
                            [1, 0], [0, 1], [-1, 0], [0, -1],
                            [1, 1], [1, -1], [-1, 1], [-1, -1],
                            ], device=targets_6.device).float() * g

        for i in range(self.nl):
            anchors = self.anchors[i]
            gain[2:6] = torch.tensor(p[i].shape)[[3, 2, 3, 2]]
                
            # Match targets_6 to anchors
            t = targets_6 * gain
            if nt:
                r = t[:, :, 4:6] / anchors[:, None] # wh ratio
                j = torch.max(r, 1. / r).max(2)[0] < self.hyp['anchor_t'] # 
                t = t[j] # [3, 33, 7] -> [59 (true_j), 7]

                # Offsets
                # This part of the code is determining which quadrant of the grid cell the center of each target object falls into. 
                # j and k check for the upper left and lower right quadrants respectively, 
                # while l and m check for the upper right and lower left quadrants respectively.
                gxy = t[:, 2:4]
                gxi = gain[[2, 3]] - gxy
                j, k = ((gxy % 1. < g) & (gxy > 1.)).T
                l, m = ((gxi % 1. < g) & (gxi > 1.)).T
                # j = torch.stack((torch.ones_like(j), j, k, l, m))
                j = torch.stack([torch.ones_like(j)] * 9, dim=-1)
                # # Filtering t such that we get copies of t that correspond to the quadrants where the object's center falls into.
                # t = t.repeat((5, 1, 1))[j]
                t = t.repeat((9, 1, 1))[j]
                offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]
            else:
                t = targets_6[0]
                offsets = 0

            b, c = t[:, :2].long().T
            gxy = t[:, 2:4]
            gwh = t[:, 4:6]
            gij = (gxy - offsets).long()
            gi, gj = gij.T

            a = t[:, 6].long()
            indices.append((b, a, gj.clamp_(0, gain[3] - 1), gi.clamp_(0, gain[2] - 1)))

            anch.append(anchors[a])

        return indices, anch

if __name__ == '__main__':

    # Example values (these are random for illustration purposes)
    batch_size = 8
    num_anchors = 3
    num_targets = 10
    num_class_cloth = 13
    num_class_action = 80
    num_classes = num_class_cloth + num_class_action
    
    # Calculate the number of elements for each anchor
    elements_per_anchor = 4 + 1 + num_classes

    def calculate_predictions(h, w):
        
        
        # Calculate the number of grid cells in the height and width directions
        grid_h, grid_w = h, w

        # Initialize the predictions tensor with zeros
        predictions = torch.zeros(batch_size, grid_h, grid_w, num_anchors * elements_per_anchor)

        # Fill in the values for each anchor
        for b in range(batch_size):
            confidence_scores = torch.rand(batch_size, h, w, num_anchors)
            bounding_boxes = torch.rand(batch_size, h, w, num_anchors, 4)
            class_probs = torch.rand(batch_size, h, w, num_anchors, num_classes)
            for y in range(grid_h):
                for x in range(grid_w):
                    for a in range(num_anchors):
                        anchor_offset = a * elements_per_anchor

                        # Confidence score
                        predictions[b, y, x, anchor_offset : anchor_offset + 4] = bounding_boxes[b, y, x, a] 

                        # Bounding box coordinates (xywh)
                        predictions[b, y, x, anchor_offset + 4 : anchor_offset + 5] = confidence_scores[b, y, x, a]

                        # Class probabilities
                        predictions[b, y, x, anchor_offset + 5 : anchor_offset + 5 + num_classes] = class_probs[b, y, x, a]

        predictions = predictions.view(batch_size, num_anchors, grid_h, grid_w, elements_per_anchor)
        
        return predictions
    
    # Predictions for image size (7, 7)
    p0 = calculate_predictions(7, 7)

    # Predictions for image size (14, 14)
    p1 = calculate_predictions(14, 14)

    # Predictions for image size (28, 28)
    p2 = calculate_predictions(28, 28)

    print('Shape of dummy predictions: ',p0.shape)  # Output: torch.Size([8, 3, 7, 7, (4+1+13+80)]) (assuming 3 anchors and 13+80 classes)
    
    # Calculate the number of elements for each anchor
    elements_per_anchor = 1 + 4 + num_classes
    
    # Initialize the targets tensor with zeros
    targets = torch.zeros(num_targets, num_anchors * elements_per_anchor)
    
    # Fill in the values for each anchor
    for b in range(num_targets):
        batch_index = b % batch_size  # Index of the batch that the target belongs to
        bounding_boxes = torch.rand(num_targets, num_anchors * 4)
        class_probs = torch.rand(num_targets, num_anchors * num_classes).round()

        for a in range(num_anchors):
            anchor_offset = a * elements_per_anchor

            # batch_index
            targets[b, anchor_offset] = batch_index

            # Bounding box coordinates (xywh)
            targets[b, anchor_offset + 1 : anchor_offset + 5] = bounding_boxes[b, a]

            # Class probabilities
            targets[b, anchor_offset + 5 : anchor_offset + 5 + num_classes] = class_probs[b, a]

    print('Shape of dummy targets: ',targets.shape)  # Output: torch.Size([10, 3*(1+4+13+80)]) (assuming 3 anchors and 13+80 classes)
    
    L = CustomLoss_()
    L.build_targets([p0, p1, p2], targets)
    loss1 = L.loss_class_cloth([p0, p1, p2], targets)
    print('Loss of cloth: ', loss1)
    
    
    