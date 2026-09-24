"""Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

import torch 
import torch.nn as nn 
import torch.nn.functional as F 

import torchvision

from ...core import register
from .box_ops import predicted_class_max_iou


__all__ = ['RTDETRPostProcessor']


def mod(a, b):
    out = a - a // b * b
    return out


@register()
class RTDETRPostProcessor(nn.Module):
    __share__ = [
        'num_classes', 
        'use_focal_loss', 
        'num_top_queries', 
        'remap_mscoco_category'
    ]
    
    def __init__(
        self, 
        num_classes=80, 
        use_focal_loss=True, 
        num_top_queries=300, 
        remap_mscoco_category=False,
        final_quality_gamma=0.0,
        oracle_final_iou_gamma=0.0,
        class_aware_oracle_final_iou_gamma=0.0,
        final_score_method='default',
        decoder_quality_beta=1.0,
        decoder_quality_iou_mode='class_agnostic',
    ) -> None:
        super().__init__()
        self.use_focal_loss = use_focal_loss
        self.num_top_queries = num_top_queries
        self.num_classes = int(num_classes)
        self.remap_mscoco_category = remap_mscoco_category 
        self.final_quality_gamma = float(final_quality_gamma)
        self.oracle_final_iou_gamma = float(oracle_final_iou_gamma)
        self.class_aware_oracle_final_iou_gamma = float(
            class_aware_oracle_final_iou_gamma)
        if final_score_method not in ('default', 'decoder_quality'):
            raise ValueError(
                f'Unsupported final_score_method: {final_score_method}')
        self.final_score_method = final_score_method
        self.decoder_quality_beta = float(decoder_quality_beta)
        if decoder_quality_iou_mode not in (
                'class_agnostic', 'predicted_class'):
            raise ValueError(
                'decoder_quality_iou_mode must be class_agnostic or '
                'predicted_class')
        self.decoder_quality_iou_mode = decoder_quality_iou_mode
        active_rerankers = sum(gamma != 0.0 for gamma in (
            self.final_quality_gamma,
            self.oracle_final_iou_gamma,
            self.class_aware_oracle_final_iou_gamma,
        ))
        if (self.final_score_method == 'decoder_quality'
                and self.decoder_quality_beta != 0.0):
            active_rerankers += 1
        if active_rerankers > 1:
            raise ValueError(
                'Learned-quality and oracle re-ranking modes are mutually '
                'exclusive')
        self.deploy_mode = False 

    def extra_repr(self) -> str:
        return (f'use_focal_loss={self.use_focal_loss}, '
                f'num_classes={self.num_classes}, '
                f'num_top_queries={self.num_top_queries}, '
                f'final_quality_gamma={self.final_quality_gamma}, '
                f'oracle_final_iou_gamma={self.oracle_final_iou_gamma}, '
                'class_aware_oracle_final_iou_gamma='
                f'{self.class_aware_oracle_final_iou_gamma}, '
                f'final_score_method={self.final_score_method}, '
                f'decoder_quality_beta={self.decoder_quality_beta}, '
                f'decoder_quality_iou_mode={self.decoder_quality_iou_mode}')
    
    # def forward(self, outputs, orig_target_sizes):
    def forward(self, outputs, orig_target_sizes: torch.Tensor,
                targets=None, image_hw=None):
        logits, boxes = outputs['pred_logits'], outputs['pred_boxes']
        # orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)        

        bbox_pred = torchvision.ops.box_convert(boxes, in_fmt='cxcywh', out_fmt='xyxy')
        oracle_quality = None
        class_aware_oracle_quality = None
        oracle_enabled = (
            self.oracle_final_iou_gamma != 0.0
            or self.class_aware_oracle_final_iou_gamma != 0.0)
        if oracle_enabled:
            if not self.use_focal_loss:
                raise ValueError(
                    'Oracle final-IoU re-ranking requires focal-loss scores')
            if targets is None or image_hw is None:
                raise ValueError(
                    'Oracle final-IoU re-ranking requires targets and image_hw')
            if len(targets) != bbox_pred.shape[0]:
                raise ValueError('Oracle target batch size does not match outputs')

            image_h, image_w = image_hw
            input_scale = bbox_pred.new_tensor(
                [image_w, image_h, image_w, image_h])
            if self.oracle_final_iou_gamma != 0.0:
                # Existing query-class oracle: one quality per class score.
                oracle_quality = logits.new_zeros(logits.shape)
            else:
                # Strict requested oracle: one predicted-class quality per query.
                class_aware_oracle_quality = logits.new_zeros(
                    (*logits.shape[:2], 1))
            for batch_index, target in enumerate(targets):
                target_boxes = target['boxes'].as_subclass(torch.Tensor).to(
                    device=bbox_pred.device, dtype=bbox_pred.dtype)
                target_labels = target['labels'].as_subclass(torch.Tensor).to(
                    device=logits.device, dtype=torch.long)
                box_format = getattr(target['boxes'], 'format', None)
                if box_format is not None and 'XYXY' not in str(box_format).upper():
                    raise ValueError(
                        'Oracle re-ranking expects validation GT in XYXY format')
                if target_boxes.numel() == 0:
                    continue
                target_boxes = target_boxes / input_scale
                ious = torchvision.ops.box_iou(
                    bbox_pred[batch_index].float(), target_boxes.float())
                if self.oracle_final_iou_gamma != 0.0:
                    for class_id in target_labels.unique().tolist():
                        if class_id < 0 or class_id >= self.num_classes:
                            raise ValueError(
                                f'GT class {class_id} is outside model class range')
                        class_mask = target_labels == class_id
                        oracle_quality[batch_index, :, class_id] = \
                            ious[:, class_mask].max(dim=1).values.to(logits.dtype)
                else:
                    class_aware_oracle_quality[batch_index, :, 0] = \
                        predicted_class_max_iou(
                            logits[batch_index].detach(),
                            bbox_pred[batch_index].detach(),
                            target_labels,
                            target_boxes).to(logits.dtype)
        bbox_pred *= orig_target_sizes.repeat(1, 2).unsqueeze(1)

        if self.use_focal_loss:
            scores = F.sigmoid(logits)
            if (self.final_score_method == 'decoder_quality'
                    and self.decoder_quality_beta != 0.0):
                if 'pred_quality_logits' not in outputs:
                    raise KeyError(
                        'decoder_quality scoring requires '
                        'pred_quality_logits')
                decoder_quality_logits = outputs['pred_quality_logits']
                if decoder_quality_logits.shape != (*scores.shape[:2], 1):
                    raise ValueError(
                        'pred_quality_logits must have shape [B, Q, 1] '
                        'matching decoder predictions')
                decoder_quality = decoder_quality_logits.sigmoid()
                scores = scores * decoder_quality.pow(
                    self.decoder_quality_beta)
            elif self.final_quality_gamma != 0.0:
                if 'enc_topk_quality_logits' not in outputs:
                    raise KeyError(
                        'final_quality_gamma requires enc_topk_quality_logits')
                quality_logits = outputs['enc_topk_quality_logits']
                if quality_logits.shape != (*scores.shape[:2], 1):
                    raise ValueError(
                        'enc_topk_quality_logits must have shape [B, Q, 1] '
                        'matching decoder predictions')
                query_quality = quality_logits.sigmoid()
                scores = scores * query_quality.pow(self.final_quality_gamma)
            elif self.oracle_final_iou_gamma != 0.0:
                scores = scores * oracle_quality.pow(
                    self.oracle_final_iou_gamma)
            elif self.class_aware_oracle_final_iou_gamma != 0.0:
                scores = scores * class_aware_oracle_quality.pow(
                    self.class_aware_oracle_final_iou_gamma)
            scores, index = torch.topk(scores.flatten(1), self.num_top_queries, dim=-1)
            # TODO for older tensorrt
            # labels = index % self.num_classes
            labels = mod(index, self.num_classes)
            index = index // self.num_classes
            boxes = bbox_pred.gather(dim=1, index=index.unsqueeze(-1).repeat(1, 1, bbox_pred.shape[-1]))
            
        else:
            scores = F.softmax(logits, dim=-1)[:, :, :-1]
            scores, labels = scores.max(dim=-1)
            boxes = bbox_pred
            if scores.shape[1] > self.num_top_queries:
                scores, index = torch.topk(scores, self.num_top_queries, dim=-1)
                labels = torch.gather(labels, dim=1, index=index)
                boxes = torch.gather(boxes, dim=1, index=index.unsqueeze(-1).tile(1, 1, boxes.shape[-1]))
        
        # TODO for onnx export
        if self.deploy_mode:
            return labels, boxes, scores

        # TODO
        if self.remap_mscoco_category:
            from ...data.dataset import mscoco_label2category
            labels = torch.tensor([mscoco_label2category[int(x.item())] for x in labels.flatten()])\
                .to(boxes.device).reshape(labels.shape)

        results = []
        for lab, box, sco in zip(labels, boxes, scores):
            result = dict(labels=lab, boxes=box, scores=sco)
            results.append(result)
        
        return results
        

    def deploy(self, ):
        self.eval()
        self.deploy_mode = True
        return self 
