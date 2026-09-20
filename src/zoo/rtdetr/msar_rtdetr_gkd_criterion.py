"""RT-DETRv2 final-query global semantic KD criterion for MSAR."""
import math

import torch
import torch.nn.functional as F

from ...core import register
from .rtdetrv2_criterion import RTDETRCriterionv2


def aggregate_query_logits(pred_logits, tau_q=1.0):
    q = pred_logits.shape[1]
    return tau_q * (torch.logsumexp(pred_logits / tau_q, dim=1) - math.log(q))


def global_semantic_kd_loss(student_score, teacher_score, eps=1e-6):
    teacher = teacher_score.detach()
    teacher = F.normalize(teacher - teacher.mean(dim=-1, keepdim=True), p=2, dim=-1, eps=eps)
    student = F.normalize(student_score - student_score.mean(dim=-1, keepdim=True), p=2, dim=-1, eps=eps)
    return (1.0 - (teacher * student).sum(dim=-1)).mean()


@register()
class MSARRTDETRCriterionv2(RTDETRCriterionv2):
    """Base RT-DETRv2 losses plus one final-output-only global semantic KD term."""

    __share__ = ["num_classes"]

    def __init__(self, matcher, weight_dict, losses, alpha=0.2, gamma=2.0,
                 num_classes=80, boxes_weight_format=None, share_matched_indices=False,
                 kd_weight=0.5, tau_q=1.0, grad_norm_interval=0):
        super().__init__(matcher, weight_dict, losses, alpha, gamma, num_classes,
                         boxes_weight_format, share_matched_indices)
        self.kd_weight = float(kd_weight)
        self.tau_q = float(tau_q)
        self.grad_norm_interval = int(grad_norm_interval)
        if self.grad_norm_interval < 0:
            raise ValueError("grad_norm_interval must be non-negative")
        self._sanity_printed = False

    def forward(self, outputs, targets, **kwargs):
        losses = super().forward(outputs, targets, **kwargs)
        pred_logits = outputs["pred_logits"]  # final decoder layer only
        teacher_score = torch.stack(
            [target["teacher_global_score"] for target in targets], dim=0
        ).to(device=pred_logits.device, dtype=torch.float32)
        student_score = aggregate_query_logits(pred_logits.float(), self.tau_q)
        loss_raw = global_semantic_kd_loss(student_score, teacher_score)
        losses["loss_gkd"] = self.kd_weight * loss_raw
        # Metric-only key; det_engine excludes *_raw from the optimized sum.
        losses["loss_gkd_raw"] = loss_raw.detach()
        if not self._sanity_printed:
            print(
                "GKD_SANITY "
                f"pred_logits={tuple(pred_logits.shape)} "
                f"teacher_score={tuple(teacher_score.shape)} "
                f"student_score={tuple(student_score.shape)} "
                f"loss_gkd_raw={loss_raw.item():.6f} "
                f"weighted_gkd={losses['loss_gkd'].item():.6f}",
                flush=True,
            )
            self._sanity_printed = True
        return losses
