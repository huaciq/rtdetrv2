"""Local SAR structure for the encoder localization-quality branch only."""

import torch
from torch import nn
from torch.nn import functional as F


class SARQualityHead(nn.Module):
    """Predict quality from [f, f - AvgPool(f), detached proposal cxcywh].

    Pool each feature level independently in spatial coordinates. A single
    affine fusion keeps the E1 linear quality predictor lightweight; neither
    candidate features nor proposals used by the detector are replaced.
    """

    def __init__(self, hidden_dim, pool_size=3):
        super().__init__()
        if not isinstance(pool_size, int) or pool_size < 3 or pool_size % 2 != 1:
            raise ValueError('sar_quality_pool_size must be an odd integer >= 3')
        self.pool_size = pool_size
        self.proj = nn.Linear(2 * hidden_dim + 4, 1)
        # Same initial constant quality (0.5) as QAQS.
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def structural_residual(self, features, spatial_shapes):
        sizes = [h * w for h, w in spatial_shapes]
        levels = features.split(sizes, dim=1)
        residuals = []
        for level, (h, w) in zip(levels, spatial_shapes):
            grid = level.transpose(1, 2).reshape(features.shape[0], -1, h, w)
            smooth = F.avg_pool2d(
                grid, self.pool_size, stride=1,
                padding=self.pool_size // 2, count_include_pad=False)
            residuals.append((grid - smooth).flatten(2).transpose(1, 2))
        return torch.cat(residuals, dim=1)

    def forward(self, features, proposal_logits, spatial_shapes):
        residual = self.structural_residual(features, spatial_shapes)
        # Quality supervision must not create a new bbox-head gradient path.
        geometry = proposal_logits.detach().sigmoid().to(features.dtype)
        return self.proj(torch.cat((features, residual, geometry), dim=-1))
