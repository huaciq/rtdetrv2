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
            scalar_probe = getattr(decoder, 'final_quality_probe', False)
            class_probe = getattr(
                decoder, 'class_conditioned_final_quality_probe', False)
            multi_threshold_probe = getattr(
                decoder,
                'multi_threshold_class_conditioned_quality_probe', False)
            if not (scalar_probe or class_probe or multi_threshold_probe):
                raise ValueError(
                    'freeze_detector_for_final_quality_probe requires '
                    'a final quality probe in RTDETRTransformerv2')
            if scalar_probe:
                probe_head = decoder.final_quality_probe_head
            elif class_probe:
                probe_head = decoder.class_conditioned_quality_probe_head
            else:
                probe_head = decoder.multi_threshold_quality_probe_head
            for parameter in self.parameters():
                parameter.requires_grad_(False)
            for parameter in probe_head.parameters():
                parameter.requires_grad_(True)

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_detector_for_final_quality_probe:
            # Frozen detector modules stay in inference mode so BatchNorm
            # buffers and inference-time decoder behavior are also unchanged.
            self.backbone.eval()
            self.encoder.eval()
            self.decoder.eval()
            if getattr(self.decoder, 'final_quality_probe', False):
                self.decoder.final_quality_probe_head.train(mode)
            elif getattr(
                    self.decoder,
                    'class_conditioned_final_quality_probe', False):
                self.decoder.class_conditioned_quality_probe_head.train(mode)
            else:
                self.decoder.multi_threshold_quality_probe_head.train(mode)
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
