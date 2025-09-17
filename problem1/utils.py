import torch
import numpy as np

def generate_anchors(feature_map_sizes, anchor_scales, image_size=224):
    """
    Generate anchors for multiple feature maps.

    Args:
        feature_map_sizes: List of (H, W) tuples for each feature map
        anchor_scales: List of lists, scales for each feature map
        image_size: Input image size

    Returns:
        anchors: List of tensors, each of shape [H*W*num_anchors, 4]
                 in [x1, y1, x2, y2] format
    """
    # For each feature map:
    # 1. Create grid of anchor centers
    # 2. Generate anchors with specified scales and ratios
    # 3. Convert to absolute coordinates
    anchors_per_scale = []

    for (height, width), scales in zip(feature_map_sizes, anchor_scales):
        stride = image_size / height
        y_centers = torch.arange(height) * stride + stride / 2
        x_centers = torch.arange(width) * stride + stride / 2
        yy, xx = torch.meshgrid(y_centers, x_centers, indexing='ij')
        centers = torch.stack([xx, yy], dim=-1).view(-1, 2)  # [H*W, 2]
        num_locations = centers.shape[0]
        anchors = []
        for scale in scales:
            w = h = scale
            wh = torch.tensor([w, h], dtype=torch.float32)
            anchor = torch.cat([centers-wh/2, centers+wh/2], dim=1)
            anchors.append(anchor)
        anchors = torch.cat(anchors, dim=0)
        anchors_per_scale.append(anchors)

    return anchors_per_scale
    pass

def compute_iou(boxes1, boxes2):
    """
    Compute IoU between two sets of boxes.

    Args:
        boxes1: Tensor of shape [N, 4]
        boxes2: Tensor of shape [M, 4]

    Returns:
        iou: Tensor of shape [N, M]
    """
    N = boxes1.size(0)
    M = boxes2.size(0)
    boxes1 = boxes1[:, None, :]
    boxes2 = boxes2[None, :, :]
    top_left = torch.max(boxes1[..., :2], boxes2[..., :2])
    bottom_right = torch.min(boxes1[..., 2:], boxes2[..., 2:])
    wh = (bottom_right - top_left).clamp(min=0)
    area = wh[..., 0] * wh[..., 1]
    area1 = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
    area2 = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])
    union_area = area1 + area2 - area
    iou = area / (union_area + 1e-6)
    return iou
    pass

def match_anchors_to_targets(anchors, target_boxes, target_labels,
                            pos_threshold=0.5, neg_threshold=0.3):
    """
    Match anchors to ground truth boxes.

    Args:
        anchors: Tensor of shape [num_anchors, 4]
        target_boxes: Tensor of shape [num_targets, 4]
        target_labels: Tensor of shape [num_targets]
        pos_threshold: IoU threshold for positive anchors
        neg_threshold: IoU threshold for negative anchors

    Returns:
        matched_labels: Tensor of shape [num_anchors]
                       (0: background, 1-N: classes)
        matched_boxes: Tensor of shape [num_anchors, 4]
        pos_mask: Boolean tensor indicating positive anchors
        neg_mask: Boolean tensor indicating negative anchors
    """
    device = anchors.device
    target_boxes = target_boxes.to(device)
    target_labels = target_labels.to(device)
    num_anchors = anchors.size(0)
    matched_labels = torch.zeros(num_anchors, dtype=torch.long, device=device)
    matched_boxes = torch.zeros_like(anchors, device=device)
    iou = compute_iou(anchors, target_boxes)
    max_iou, max_indices = iou.max(dim=1)
    pos_mask = max_iou >= pos_threshold
    neg_mask = max_iou < neg_threshold
    matched_boxes[pos_mask] = target_boxes[max_indices[pos_mask]]
    matched_labels[pos_mask] = target_labels[max_indices[pos_mask]] + 1
    best_anchor_indices = iou.argmax(dim=0)
    matched_boxes[best_anchor_indices] = target_boxes
    matched_labels[best_anchor_indices] = target_labels + 1
    pos_mask[best_anchor_indices] = True
    neg_mask[best_anchor_indices] = False
    return matched_labels, matched_boxes, pos_mask, neg_mask
    pass
