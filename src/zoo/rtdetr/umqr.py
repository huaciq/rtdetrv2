"""Shared five-keypoint/uncertainty geometry for decoder box refinement."""

import math

import torch
from torch import nn

from .box_ops import box_cxcywh_to_xyxy
from .utils import inverse_sigmoid


def box_keypoints(boxes):
    """Normalized cxcywh -> [center, left, right, top, bottom], [..., 5, 2].

    Training transforms already normalize the GT boxes; no new annotations,
    image-size convention or keypoint definition are introduced here.
    """
    x1, y1, x2, y2 = box_cxcywh_to_xyxy(boxes).unbind(-1)
    cx, cy = (x1 + x2) * 0.5, (y1 + y2) * 0.5
    return torch.stack([
        torch.stack([cx, cy], -1), torch.stack([x1, cy], -1),
        torch.stack([x2, cy], -1), torch.stack([cx, y1], -1),
        torch.stack([cx, y2], -1),
    ], dim=-2)


def validate_log_scale_bounds(lower, upper):
    if not (math.isfinite(lower) and math.isfinite(upper)
            and -10 <= lower < upper <= 10):
        raise ValueError('UMQR log-scale bounds must satisfy -10 <= min < max <= 10')


def keypoints_to_box(points, eps=1e-5):
    """Five points -> valid normalized cxcywh; fixed center/midpoint mean.

    Crossed edges get a positive minimum size. Constrain the center using
    that size so the resulting xyxy corners also remain within the image.
    """
    points = points.float().clamp(0., 1.)
    left, right = points[..., 1, 0], points[..., 2, 0]
    top, bottom = points[..., 3, 1], points[..., 4, 1]
    size = torch.stack((right - left, bottom - top), -1).clamp(eps, 1. - 2 * eps)
    edge_center = torch.stack(((left + right) * .5, (top + bottom) * .5), -1)
    center = (points[..., 0, :] + edge_center) * .5
    center = torch.maximum(size * .5 + eps, torch.minimum(center, 1. - size * .5 - eps))
    return torch.cat((center, size), -1)


class UMQRHead(nn.Module):
    """Shared five-point/scales head with hidden or direct bbox guidance."""

    def __init__(self, hidden_dim, bottleneck_dim=64,
                 log_scale_min=-5.0, log_scale_max=3.0, direct_box=False):
        super().__init__()
        validate_log_scale_bounds(log_scale_min, log_scale_max)
        if not isinstance(bottleneck_dim, int) or bottleneck_dim <= 0:
            raise ValueError('umqr_hidden_dim must be a positive integer')
        self.log_scale_min = log_scale_min
        self.log_scale_max = log_scale_max
        self.direct_box = direct_box
        self.prediction = nn.Sequential(
            nn.Linear(hidden_dim, bottleneck_dim), nn.ReLU(),
            nn.Linear(bottleneck_dim, 20))
        if direct_box:
            self.alpha = nn.Parameter(torch.zeros(()))
        else:
            self.geometry = nn.Sequential(
                nn.Linear(20, bottleneck_dim), nn.ReLU(),
                nn.Linear(bottleneck_dim, hidden_dim))
        # Start at the current proposal's five-point template, sigma=1.
        nn.init.zeros_(self.prediction[-1].weight)
        nn.init.zeros_(self.prediction[-1].bias)
        # Small residual with a live geometry gradient path once bbox heads
        # leave the baseline's zero-initialized final projection.
        if not direct_box:
            nn.init.normal_(self.geometry[-1].weight, std=1e-3)
            nn.init.zeros_(self.geometry[-1].bias)

    def forward(self, features, reference_boxes):
        prediction = self.prediction(features)
        # FP32 sigmoid/reference logits are stable even under detector AMP.
        with torch.autocast(device_type=features.device.type, enabled=False):
            prediction = prediction.float()
            template = box_keypoints(reference_boxes.detach().float())
            offsets = prediction[..., :10].reshape(*features.shape[:-1], 5, 2)
            points = (inverse_sigmoid(template, eps=1e-5) + offsets).sigmoid()
            log_scales = prediction[..., 10:].reshape_as(points).clamp(
                self.log_scale_min, self.log_scale_max)
            geometry_input = torch.cat((points.flatten(-2), log_scales.flatten(-2)), -1)
        if self.direct_box:
            return features, points, log_scales
        guidance = self.geometry(geometry_input.to(features.dtype))
        return features + guidance, points, log_scales

    def box_residual(self, points, log_scales, reference_boxes):
        """alpha*C_geo*(logit(B_kp)-logit(reference)), all geometry in FP32."""
        with torch.autocast(device_type=points.device.type, enabled=False):
            boxes = keypoints_to_box(points)
            scales = log_scales.float().clamp(self.log_scale_min, self.log_scale_max)
            confidence = (-scales.mean(dim=(-2, -1))).sigmoid().unsqueeze(-1)
            delta = inverse_sigmoid(boxes) - inverse_sigmoid(reference_boxes.detach().float())
            residual = self.alpha.float() * confidence * delta
        return residual, boxes, confidence


def uncertainty_keypoint_loss(points, log_scales, target_points, num_boxes,
                              log_scale_min=-5.0, log_scale_max=3.0):
    """Mean over 5x2 coordinates, sum foreground objects / DDP-average GT count.

    Empty foreground selections produce a differentiable zero for both
    predictions. Negative values are valid Laplace log-scale NLL values.
    """
    with torch.autocast(device_type=points.device.type, enabled=False):
        scales = log_scales.float().clamp(log_scale_min, log_scale_max)
        error = (points.float() - target_points.float()).abs()
        terms = torch.exp(-scales) * error + scales
        return terms.mean(dim=(-2, -1)).sum() / num_boxes
