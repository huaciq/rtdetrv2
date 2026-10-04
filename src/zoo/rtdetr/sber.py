"""E4: fixed boundary sampling of actual encoder maps; box-only residual."""

import math

import torch
from torch import nn
from torch.nn import functional as F

from .box_ops import box_cxcywh_to_xyxy, box_xyxy_to_cxcywh


def encoder_feature_maps(memory, spatial_shapes):
    """Views of the unchanged projected HybridEncoder P3/P4/P5 memory."""
    sizes = [height * width for height, width in spatial_shapes]
    return [part.transpose(1, 2).reshape(memory.shape[0], memory.shape[-1], height, width).float()
            for part, (height, width) in zip(memory.split(sizes, dim=1), spatial_shapes)]


def boundary_sampling_coordinates(reference_boxes, rho=0.1):
    """[B,Q,4,3,2]: left/right/top/bottom x inside/boundary/outside."""
    boxes = reference_boxes.detach().float().clamp(0., 1.)
    center, size = boxes[..., :2], boxes[..., 2:].clamp_min(1e-6)
    cx, cy = center.unbind(-1)
    width, height = size.unbind(-1)
    boundary = torch.stack((
        torch.stack((cx - width * .5, cy), -1),
        torch.stack((cx + width * .5, cy), -1),
        torch.stack((cx, cy - height * .5), -1),
        torch.stack((cx, cy + height * .5), -1)), -2)
    zero = torch.zeros_like(width)
    inward = rho * torch.stack((
        torch.stack((width, zero), -1),
        torch.stack((-width, zero), -1),
        torch.stack((zero, height), -1),
        torch.stack((zero, -height), -1)), -2)
    return torch.stack((boundary + inward, boundary, boundary - inward), -2).clamp(0., 1.)


def box_scale_weights(reference_boxes, spatial_shapes):
    """Fixed size-based interpolation; four P3 cells correspond to level 0."""
    height, width = spatial_shapes[0]
    size = reference_boxes.detach().float()[..., 2:].clamp_min(1e-6)
    p3_extent = (size[..., 0] * width * size[..., 1] * height).sqrt()
    level = (p3_extent / 4.).clamp_min(1e-6).log2().clamp(0., len(spatial_shapes) - 1)
    indices = torch.arange(len(spatial_shapes), device=level.device, dtype=level.dtype)
    weights = (1. - (level.unsqueeze(-1) - indices).abs()).clamp_min(0.)
    return weights / weights.sum(-1, keepdim=True).clamp_min(1e-6)


def sample_boundary_features(feature_maps, coordinates, weights):
    """Bilinear reads from each real map, not from the query feature."""
    batch, queries = coordinates.shape[:2]
    grid = coordinates.reshape(batch, queries * 4, 3, 2) * 2. - 1.
    mixed = None
    for level, feature in enumerate(feature_maps):
        sampled = F.grid_sample(feature.float(), grid, mode='bilinear',
                                padding_mode='border', align_corners=False)
        sampled = sampled.permute(0, 2, 3, 1).reshape(batch, queries, 4, 3, feature.shape[1])
        contribution = sampled * weights[..., level, None, None, None]
        mixed = contribution if mixed is None else mixed + contribution
    return mixed


def apply_boundary_residual(box_det, fractions, eps=1e-6):
    """Bounded edge shifts of the original bbox result, then valid cxcywh.

    Fractions are [dl,dr,dt,db], measured in box_det width/height. They are
    <=rho in magnitude, so rho<.5 prevents crossing before image clipping.
    Projection to legal image corners handles baseline boxes at the border.
    """
    boxes = box_det.float()
    width, height = boxes[..., 2:].unbind(-1)
    dl, dr, dt, db = fractions.unbind(-1)
    shift_xyxy = torch.stack((dl * width, dt * height, dr * width, db * height), -1)
    corners = box_cxcywh_to_xyxy(boxes) + shift_xyxy
    lower = corners[..., :2].clamp(0., 1. - eps)
    upper = torch.maximum(corners[..., 2:].clamp(eps, 1.), lower + eps)
    refined = box_xyxy_to_cxcywh(torch.cat((lower, upper), -1))
    return refined


class SBERHead(nn.Module):
    """One small MLP shared across all layers, trained by detection losses."""

    def __init__(self, channels, hidden_dim=64, rho=0.1):
        super().__init__()
        if not math.isfinite(rho) or not 0 < rho < .5:
            raise ValueError('sber_rho must be finite and in (0,.5)')
        if not isinstance(hidden_dim, int) or hidden_dim <= 0:
            raise ValueError('sber_hidden_dim must be a positive integer')
        self.rho = rho
        self.evidence_mlp = nn.Sequential(nn.Linear(8 * channels, hidden_dim), nn.ReLU())
        self.offset_head = nn.Linear(hidden_dim, 4)
        self.gate_head = nn.Linear(hidden_dim, 1)
        # Residual starts at zero; gate starts at .5. No new loss is needed.
        nn.init.zeros_(self.offset_head.weight)
        nn.init.zeros_(self.offset_head.bias)
        nn.init.zeros_(self.gate_head.weight)
        nn.init.zeros_(self.gate_head.bias)

    def forward(self, feature_maps, reference_boxes, collect_debug=False):
        with torch.autocast(device_type=reference_boxes.device.type, enabled=False):
            coordinates = boundary_sampling_coordinates(reference_boxes, self.rho)
            shapes = [feature.shape[-2:] for feature in feature_maps]
            weights = box_scale_weights(reference_boxes, shapes)
            sampled = sample_boundary_features(feature_maps, coordinates, weights)
            contrast = sampled[..., 0, :] - sampled[..., 2, :]
            boundary = sampled[..., 1, :]
            evidence = torch.cat((contrast.flatten(-2), boundary.flatten(-2)), -1)
            hidden = self.evidence_mlp(evidence)
            gate_logits = self.gate_head(hidden)
            offset_logits = self.offset_head(hidden)
            gate = gate_logits.clamp(-10., 10.).sigmoid()
            fractions = offset_logits.tanh() * self.rho * gate
            details = None
            if collect_debug:
                # Scalar summaries only; never retain batch-sized debug maps.
                details = {
                    'sampling_min': coordinates.detach().min(),
                    'sampling_max': coordinates.detach().max(),
                    'contrast_abs_mean': contrast.detach().abs().mean(),
                    'inside_outside_different_fraction': (contrast.detach().abs() > 1e-6).float().mean(),
                    'gate_mean': gate.detach().mean(),
                    'gate_min': gate.detach().min(),
                    'gate_max': gate.detach().max(),
                    'offset_fraction_abs_max': fractions.detach().abs().max(),
                    'scale_weight_mean': weights.detach().mean(dim=(0, 1)),
                    'evidence_finite': (torch.isfinite(sampled.detach()).all()
                                        & torch.isfinite(hidden.detach()).all()
                                        & torch.isfinite(gate_logits.detach()).all()
                                        & torch.isfinite(offset_logits.detach()).all()),
                }
        return fractions, details
