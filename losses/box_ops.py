"""
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
https://github.com/facebookresearch/detr/blob/main/util/box_ops.py
"""

import torch
import math
from torch import Tensor
from torchvision.ops.boxes import box_area


def box_cxcywh_to_xyxy(x: Tensor) -> Tensor:
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)


def box_xyxy_to_cxcywh(x: Tensor) -> Tensor:
    x0, y0, x1, y1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2,
         (x1 - x0), (y1 - y0)]
    return torch.stack(b, dim=-1)


# modified from torchvision to also return the union
def box_iou(boxes1: Tensor, boxes2: Tensor):
    # Force float32 for numerical stability under AMP
    orig_dtype = boxes1.dtype
    boxes1 = boxes1.float()
    boxes2 = boxes2.float()
    
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter
    union = union.clamp(min=1e-6)

    iou = inter / union
    iou = iou.clamp(min=0.0, max=1.0)
    
    return iou.to(orig_dtype), union.to(orig_dtype)


def generalized_box_iou(boxes1, boxes2):
    """
    Generalized IoU from https://giou.stanford.edu/

    The boxes should be in [x0, y0, x1, y1] format

    Returns a [N, M] pairwise matrix, where N = len(boxes1)
    and M = len(boxes2)
    """
    # Force float32 for numerical stability under AMP
    orig_dtype = boxes1.dtype
    boxes1 = boxes1.float()
    boxes2 = boxes2.float()
    
    # degenerate boxes gives inf / nan results
    # Handle degenerate boxes by clamping to ensure x2 >= x1, y2 >= y1
    eps = 1e-6
    boxes1 = boxes1.clone()
    boxes2 = boxes2.clone()
    # Clamp to reasonable range first
    boxes1 = boxes1.clamp(min=-10.0, max=10.0)
    boxes2 = boxes2.clamp(min=-10.0, max=10.0)
    boxes1[:, 2:] = torch.max(boxes1[:, 2:], boxes1[:, :2] + eps)
    boxes2[:, 2:] = torch.max(boxes2[:, 2:], boxes2[:, :2] + eps)
    iou, union = box_iou(boxes1, boxes2)

    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    area = wh[:, :, 0] * wh[:, :, 1]
    area = area.clamp(min=1e-6)

    result = iou - (area - union) / area
    
    # Handle NaN values
    result = torch.nan_to_num(result, nan=0.0, posinf=1.0, neginf=-1.0)
    result = result.clamp(min=-1.0, max=1.0)
    
    return result.to(orig_dtype)


def complete_box_iou(boxes1: Tensor, boxes2: Tensor):
    """
    Complete IoU (CIoU) from https://arxiv.org/abs/1911.08287
    
    boxes1, boxes2: [N, 4] in xyxy format
    Returns: [N] CIoU values (diagonal only, for matched pairs)
    """
    assert boxes1.shape[0] == boxes2.shape[0], "CIoU expects matched pairs"
    
    # Force float32 for numerical stability under AMP
    orig_dtype = boxes1.dtype
    boxes1 = boxes1.float()
    boxes2 = boxes2.float()
    
    # Clamp boxes to valid range to prevent extreme values
    boxes1 = boxes1.clamp(min=-10.0, max=10.0)
    boxes2 = boxes2.clamp(min=-10.0, max=10.0)
    
    # IoU
    iou, union = box_iou(boxes1, boxes2)
    iou = torch.diag(iou)
    
    cx1 = (boxes1[:, 0] + boxes1[:, 2]) / 2
    cy1 = (boxes1[:, 1] + boxes1[:, 3]) / 2
    cx2 = (boxes2[:, 0] + boxes2[:, 2]) / 2
    cy2 = (boxes2[:, 1] + boxes2[:, 3]) / 2
    center_dist_sq = (cx1 - cx2) ** 2 + (cy1 - cy2) ** 2
    
    enclose_x1 = torch.min(boxes1[:, 0], boxes2[:, 0])
    enclose_y1 = torch.min(boxes1[:, 1], boxes2[:, 1])
    enclose_x2 = torch.max(boxes1[:, 2], boxes2[:, 2])
    enclose_y2 = torch.max(boxes1[:, 3], boxes2[:, 3])
    enclose_diag_sq = (enclose_x2 - enclose_x1) ** 2 + (enclose_y2 - enclose_y1) ** 2 + 1e-6
    
    w1, h1 = boxes1[:, 2] - boxes1[:, 0], boxes1[:, 3] - boxes1[:, 1]
    w2, h2 = boxes2[:, 2] - boxes2[:, 0], boxes2[:, 3] - boxes2[:, 1]
    # Clamp dimensions to prevent division issues
    h1 = h1.clamp(min=1e-6)
    h2 = h2.clamp(min=1e-6)
    v = (4 / (math.pi ** 2)) * ((torch.atan(w2 / h2) - torch.atan(w1 / h1)) ** 2)
    
    with torch.no_grad():
        alpha = v / (1 - iou + v + 1e-6)
    
    ciou = iou - center_dist_sq / enclose_diag_sq - alpha * v
    
    # Handle NaN and clamp
    ciou = torch.nan_to_num(ciou, nan=0.0, posinf=1.0, neginf=-1.0)
    ciou = ciou.clamp(min=-1.0, max=1.0)
    
    return ciou.to(orig_dtype)


def normalized_wasserstein_distance(boxes1: Tensor, boxes2: Tensor, tau_w: float = 0.177):
    """
    Normalized Wasserstein Distance (NWD) for small object detection.
    Treats boxes as 2D Gaussian distributions.
    
    boxes1, boxes2: [N, 4] in cxcywh format (normalized coordinates [0,1])
    tau_w: normalization constant
    
    Returns: [N] NWD similarity values in [0, 1]
    """
    # Force float32 for numerical stability
    orig_dtype = boxes1.dtype
    boxes1 = boxes1.float()
    boxes2 = boxes2.float()
    
    cx1, cy1, w1, h1 = boxes1.unbind(-1)
    cx2, cy2, w2, h2 = boxes2.unbind(-1)
    
    w2_sq = (cx1 - cx2) ** 2 + (cy1 - cy2) ** 2 + \
            0.25 * (w1 - w2) ** 2 + 0.25 * (h1 - h2) ** 2
    # Clamp to prevent overflow in sqrt
    w2_sq = w2_sq.clamp(min=0.0, max=100.0)
    w2_dist = torch.sqrt(w2_sq + 1e-6)
    
    # Clamp distance to prevent exp underflow (exp(-inf) = 0, which is fine)
    # but exp(+large) causes overflow
    w2_dist = w2_dist.clamp(max=20.0 * tau_w)
    
    nwd = torch.exp(-w2_dist / tau_w)
    nwd = nwd.clamp(min=0.0, max=1.0)
    
    return nwd.to(orig_dtype)


def sa_wiou_loss(pred_boxes: Tensor, target_boxes: Tensor, 
                 tau_area: float = 0.01, tau_w: float = 0.177):
    """
    Size-Adaptive Wasserstein-IoU Loss.
    
    Adaptively blends NWD (for small targets) and CIoU (for large targets).
    
    Args:
        pred_boxes, target_boxes: [N, 4] in cxcywh format (normalized coordinates)
        tau_area: Area threshold for small/large target transition
        tau_w: NWD normalization constant
    
    Returns: [N] SA-WIoU loss values
    """
    # Force float32 for numerical stability under AMP
    orig_dtype = pred_boxes.dtype
    pred_boxes = pred_boxes.float()
    target_boxes = target_boxes.float()
    
    gt_area = target_boxes[:, 2] * target_boxes[:, 3]
    alpha = torch.exp(-gt_area / tau_area)
    
    nwd = normalized_wasserstein_distance(pred_boxes, target_boxes, tau_w)
    nwd_loss = 1 - nwd
    
    pred_xyxy = box_cxcywh_to_xyxy(pred_boxes)
    target_xyxy = box_cxcywh_to_xyxy(target_boxes)
    ciou = complete_box_iou(pred_xyxy, target_xyxy)
    ciou_loss = 1 - ciou
    
    loss = alpha * nwd_loss + (1 - alpha) * ciou_loss
    
    # Clamp to prevent extreme values and handle NaN
    loss = torch.nan_to_num(loss, nan=0.0, posinf=10.0, neginf=0.0)
    loss = loss.clamp(min=0.0, max=10.0)
    
    return loss.to(orig_dtype)


def masks_to_boxes(masks):
    """Compute the bounding boxes around the provided masks

    The masks should be in format [N, H, W] where N is the number of masks, (H, W) are the spatial dimensions.

    Returns a [N, 4] tensors, with the boxes in xyxy format
    """
    if masks.numel() == 0:
        return torch.zeros((0, 4), device=masks.device)

    h, w = masks.shape[-2:]

    y = torch.arange(0, h, dtype=torch.float)
    x = torch.arange(0, w, dtype=torch.float)
    y, x = torch.meshgrid(y, x)

    x_mask = (masks * x.unsqueeze(0))
    x_max = x_mask.flatten(1).max(-1)[0]
    x_min = x_mask.masked_fill(~(masks.bool()), 1e8).flatten(1).min(-1)[0]

    y_mask = (masks * y.unsqueeze(0))
    y_max = y_mask.flatten(1).max(-1)[0]
    y_min = y_mask.masked_fill(~(masks.bool()), 1e8).flatten(1).min(-1)[0]

    return torch.stack([x_min, y_min, x_max, y_max], 1)