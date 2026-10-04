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


class UMQRHead(nn.Module):
    """One shared head across decoder layers; geometry changes bbox input only."""

    def __init__(self, hidden_dim, bottleneck_dim=64,
                 log_scale_min=-5.0, log_scale_max=3.0):
        super().__init__()
        validate_log_scale_bounds(log_scale_min, log_scale_max)
        if not isinstance(bottleneck_dim, int) or bottleneck_dim <= 0:
            raise ValueError('umqr_hidden_dim must be a positive integer')
        self.log_scale_min = log_scale_min
        self.log_scale_max = log_scale_max
        self.prediction = nn.Sequential(
            nn.Linear(hidden_dim, bottleneck_dim), nn.ReLU(),
            nn.Linear(bottleneck_dim, 20))
        self.geometry = nn.Sequential(
            nn.Linear(20, bottleneck_dim), nn.ReLU(),
            nn.Linear(bottleneck_dim, hidden_dim))
        # Start at the current proposal's five-point template, sigma=1.
        nn.init.zeros_(self.prediction[-1].weight)
        nn.init.zeros_(self.prediction[-1].bias)
        # Small residual with a live geometry gradient path once bbox heads
        # leave the baseline's zero-initialized final projection.
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
        guidance = self.geometry(geometry_input.to(features.dtype))
        return features + guidance, points, log_scales


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
