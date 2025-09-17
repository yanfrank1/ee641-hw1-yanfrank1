import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import match_anchors_to_targets

class DetectionLoss(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        self.num_classes = num_classes
        self.bce = nn.BCEWithLogitsLoss(reduction='none')
        self.ce = nn.CrossEntropyLoss(reduction='mean')
        self.smooth_l1 = nn.SmoothL1Loss(reduction='mean')

    def forward(self, predictions, targets, anchors):
        """
        Compute multi-task loss.

        Args:
            predictions: List of tensors from each scale
            targets: List of dicts with 'boxes' and 'labels' for each image
            anchors: List of anchor tensors for each scale

        Returns:
            loss_dict: Dict containing:
                - loss_obj: Objectness loss
                - loss_cls: Classification loss
                - loss_loc: Localization loss
                - loss_total: Weighted sum
        """
        # For each prediction scale:
        # 1. Match anchors to targets
        # 2. Compute objectness loss (BCE)
        # 3. Compute classification loss (CE) for positive anchors
        # 4. Compute localization loss (Smooth L1) for positive anchors
        # 5. Apply hard negative mining (3:1 ratio)
        total_obj_loss = 0.0
        total_cls_loss = 0.0
        total_loc_loss = 0.0
        batch_size = len(targets)

        for b in range(batch_size):
            target = targets[b]
            target_boxes = target['boxes'] / 224.0
            target_labels = target['labels']
            for scale_idx, pred in enumerate(predictions):
                B, _, H, W = pred.shape
                C = self.num_classes
                num_anchors = pred.shape[1] // (5 + C)
                pred = pred[b].view(num_anchors, 5 + C, H, W).permute(0, 2, 3, 1).contiguous()
                pred = pred.view(-1, 5 + C)
                obj_pred = pred[:, 4]
                cls_pred = pred[:, 5:]
                box_pred = pred[:, :4]
                anchor_boxes = anchors[scale_idx] / 224.0
                matched_labels, matched_boxes, pos_mask, neg_mask = match_anchors_to_targets(anchor_boxes, target_boxes, target_labels)
                encoded_boxes = torch.zeros_like(matched_boxes)
                gt_cxcy = 0.5 * (matched_boxes[:, :2] + matched_boxes[:, 2:])
                gt_wh   = matched_boxes[:, 2:] - matched_boxes[:, :2]
                anchor_cxcy = 0.5 * (anchor_boxes[:, :2] + anchor_boxes[:, 2:])
                anchor_wh   = anchor_boxes[:, 2:] - anchor_boxes[:, :2]
                encoded_boxes[:, :2] = (gt_cxcy - anchor_cxcy) / anchor_wh
                encoded_boxes[:, 2:] = torch.log(gt_wh / anchor_wh + 1e-6)
                obj_target = pos_mask.float()
                obj_loss = self.bce(obj_pred, obj_target)
                
                if pos_mask.any():
                    cls_target = matched_labels[pos_mask] - 1
                    cls_loss = self.ce(cls_pred[pos_mask], cls_target)
                    loc_loss = self.smooth_l1(box_pred[pos_mask], encoded_boxes[pos_mask]).mean()
                else:
                    cls_loss = torch.tensor(0.0, device=pred.device)
                    loc_loss = torch.tensor(0.0, device=pred.device)
                    
                neg_loss = self.bce(obj_pred, obj_target) 
                selected_neg_mask = self.hard_negative_mining(neg_loss, pos_mask, neg_mask, ratio=3)
                obj_loss = obj_loss[pos_mask].mean() + obj_loss[selected_neg_mask].mean()
                if torch.isnan(obj_loss): obj_loss = torch.tensor(0.0, device=pred.device)
                if torch.isnan(cls_loss): cls_loss = torch.tensor(0.0, device=pred.device)
                if torch.isnan(loc_loss): loc_loss = torch.tensor(0.0, device=pred.device)
                total_obj_loss += obj_loss
                total_cls_loss += cls_loss
                total_loc_loss += loc_loss

        total_obj_loss /= batch_size
        total_cls_loss /= batch_size
        total_loc_loss /= batch_size
        return {
            'loss_obj': total_obj_loss,
            'loss_cls': total_cls_loss,
            'loss_loc': total_loc_loss,
            'loss_total': total_obj_loss + total_cls_loss + total_loc_loss
        }
        pass

    def hard_negative_mining(self, loss, pos_mask, neg_mask, ratio=3):
        """
        Select hard negative examples.

        Args:
            loss: Loss values for all anchors
            pos_mask: Boolean mask for positive anchors
            neg_mask: Boolean mask for negative anchors
            ratio: Negative to positive ratio

        Returns:
            selected_neg_mask: Boolean mask for selected negatives
        """
        num_pos = pos_mask.sum().item()
        if num_pos == 0 or neg_mask.sum() == 0:
            return torch.zeros_like(neg_mask, dtype=torch.bool)
        num_neg = min(ratio * num_pos, neg_mask.sum().item())
        neg_losses = loss.clone()
        neg_losses[~neg_mask] = -1
        _, topk = torch.topk(neg_losses, int(num_neg))
        selected_neg_mask = torch.zeros_like(neg_mask)
        selected_neg_mask[topk] = True
        return selected_neg_mask
        pass
