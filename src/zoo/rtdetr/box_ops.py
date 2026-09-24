"""
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
https://github.com/facebookresearch/detr/blob/main/util/box_ops.py
"""

import torch
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
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter

    iou = inter / union
    return iou, union


def predicted_class_max_iou(
        pred_logits: Tensor,
        pred_boxes_xyxy: Tensor,
        target_labels: Tensor,
        target_boxes_xyxy: Tensor) -> Tensor:
    """Return each query's max IoU to GT of its predicted class.

    This is the shared definition used by the predicted-class-aware oracle and
    the final decoder quality probe target. Boxes must already use the same
    XYXY coordinate system.
    """
    if pred_logits.ndim != 2:
        raise ValueError('pred_logits must have shape [Q, C]')
    if pred_boxes_xyxy.shape != (pred_logits.shape[0], 4):
        raise ValueError('pred_boxes_xyxy must have shape [Q, 4]')
    if target_boxes_xyxy.ndim != 2 or target_boxes_xyxy.shape[-1] != 4:
        raise ValueError('target_boxes_xyxy must have shape [G, 4]')
    if target_labels.ndim != 1 or target_labels.shape[0] != target_boxes_xyxy.shape[0]:
        raise ValueError('target labels and boxes must have matching length')

    quality = pred_boxes_xyxy.new_zeros(pred_logits.shape[0])
    if target_boxes_xyxy.numel() == 0:
        return quality

    predicted_classes = pred_logits.argmax(dim=-1)
    ious, _ = box_iou(
        pred_boxes_xyxy.float(), target_boxes_xyxy.float())
    for class_id in predicted_classes.unique().tolist():
        target_mask = target_labels == class_id
        if not target_mask.any():
            continue
        query_mask = predicted_classes == class_id
        quality[query_mask] = ious[query_mask][:, target_mask].max(
            dim=1).values.to(quality.dtype)
    return quality


def generalized_box_iou(boxes1, boxes2):
    """
    Generalized IoU from https://giou.stanford.edu/

    The boxes should be in [x0, y0, x1, y1] format

    Returns a [N, M] pairwise matrix, where N = len(boxes1)
    and M = len(boxes2)
    """
    # degenerate boxes gives inf / nan results
    # so do an early check
    assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
    assert (boxes2[:, 2:] >= boxes2[:, :2]).all()
    iou, union = box_iou(boxes1, boxes2)

    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    area = wh[:, :, 0] * wh[:, :, 1]

    return iou - (area - union) / area


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
