"""Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

import torch 
import torch.nn as nn 
import torch.nn.functional as F 

import random 
import numpy as np 
from typing import List 

from ...core import register


__all__ = ['RTDETR', ]


@register()
class RTDETR(nn.Module):
    __inject__ = ['backbone', 'encoder', 'decoder', ]

    def __init__(self, \
        backbone: nn.Module, 
        encoder: nn.Module, 
        decoder: nn.Module, 
        freeze_detector_for_final_quality_probe: bool = False,
    ):
        super().__init__()
        self.backbone = backbone
        self.decoder = decoder
        self.encoder = encoder
        self.freeze_detector_for_final_quality_probe = \
            freeze_detector_for_final_quality_probe
        if freeze_detector_for_final_quality_probe:
            if not getattr(decoder, 'final_quality_probe', False):
                raise ValueError(
                    'freeze_detector_for_final_quality_probe requires '
                    'RTDETRTransformerv2.final_quality_probe=True')
            for parameter in self.parameters():
                parameter.requires_grad_(False)
            for parameter in decoder.final_quality_probe_head.parameters():
                parameter.requires_grad_(True)

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_detector_for_final_quality_probe:
            # Frozen detector modules stay in inference mode so BatchNorm
            # buffers and inference-time decoder behavior are also unchanged.
            self.backbone.eval()
            self.encoder.eval()
            self.decoder.eval()
            self.decoder.final_quality_probe_head.train(mode)
        return self
        
    def forward(self, x, targets=None):
        x = self.backbone(x)
        x = self.encoder(x)        
        x = self.decoder(x, targets)

        return x
    
    def deploy(self, ):
        self.eval()
        for m in self.modules():
            if hasattr(m, 'convert_to_deploy'):
                m.convert_to_deploy()
        return self 
