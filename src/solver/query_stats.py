"""Dataset-level diagnostics for RT-DETR encoder Top-K queries."""

import torch

from ..zoo.rtdetr.box_ops import box_cxcywh_to_xyxy, box_iou


class QueryStats:
    """Accumulate encoder-query coverage statistics for single-process eval."""

    IOU_THRESHOLDS = (0.1, 0.3, 0.5)
    SCORE_THRESHOLDS = (0.1, 0.3, 0.5)
    SCALE_NAMES = ('all', 'small', 'medium', 'large')

    def __init__(self):
        self.gt_counts = {name: 0 for name in self.SCALE_NAMES}
        self.best_iou_sums = {name: 0.0 for name in self.SCALE_NAMES}
        self.recalled = {
            threshold: {name: 0 for name in self.SCALE_NAMES}
            for threshold in self.IOU_THRESHOLDS
        }
        self.query_count = 0
        self.foreground = {threshold: 0 for threshold in self.IOU_THRESHOLDS}
        self.high_conf_background = {
            threshold: 0 for threshold in self.SCORE_THRESHOLDS
        }
        self.level_counts = None
        self.spatial_shapes = None

    @staticmethod
    def _scale_masks(areas):
        return {
            'all': torch.ones_like(areas, dtype=torch.bool),
            'small': areas < 32 ** 2,
            'medium': (areas >= 32 ** 2) & (areas < 96 ** 2),
            'large': areas >= 96 ** 2,
        }

    @staticmethod
    def _normalized_gt_xyxy(target, image_hw):
        boxes = target['boxes'].as_subclass(torch.Tensor).float()
        box_format = getattr(target['boxes'], 'format', None)
        if box_format is not None and 'XYXY' not in str(box_format).upper():
            raise ValueError(
                'QueryStats expects validation targets in pixel-space XYXY; '
                f'got torchvision box format {box_format}.')

        image_h, image_w = image_hw
        scale = boxes.new_tensor([image_w, image_h, image_w, image_h])
        return boxes / scale

    @staticmethod
    def _target_areas(target, normalized_gt_xyxy):
        if 'area' in target:
            return target['area'].as_subclass(torch.Tensor).float()

        # Repository datasets store orig_size as [width, height].
        width, height = target['orig_size'].tolist()
        wh = (normalized_gt_xyxy[:, 2:] - normalized_gt_xyxy[:, :2]).clamp(min=0)
        return wh[:, 0] * width * wh[:, 1] * height

    def update(self, outputs, targets, image_hw):
        required = (
            'enc_topk_boxes', 'enc_topk_logits',
            'enc_topk_indices', 'enc_spatial_shapes',
        )
        missing = [key for key in required if key not in outputs]
        if missing:
            raise KeyError(f'Missing encoder query diagnostics: {missing}')

        topk_boxes = outputs['enc_topk_boxes']
        topk_logits = outputs['enc_topk_logits']
        topk_indices = outputs['enc_topk_indices']
        spatial_shapes = outputs['enc_spatial_shapes']

        if topk_boxes.ndim != 3 or topk_boxes.shape[-1] != 4:
            raise ValueError(f'enc_topk_boxes must have shape [B, K, 4], got {topk_boxes.shape}')
        if topk_logits.shape[:2] != topk_boxes.shape[:2]:
            raise ValueError('enc_topk_logits and enc_topk_boxes must share [B, K]')
        if topk_indices.shape != topk_boxes.shape[:2]:
            raise ValueError('enc_topk_indices must have shape [B, K]')
        if spatial_shapes.ndim != 2 or spatial_shapes.shape[-1] != 2:
            raise ValueError('enc_spatial_shapes must have shape [num_levels, 2]')
        if len(targets) != topk_boxes.shape[0]:
            raise ValueError('Target batch size does not match encoder diagnostics')

        shapes = spatial_shapes.detach().cpu().long()
        if self.spatial_shapes is None:
            self.spatial_shapes = shapes
            self.level_counts = torch.zeros(len(shapes), dtype=torch.long)
        elif not torch.equal(self.spatial_shapes, shapes):
            raise ValueError(
                f'Encoder spatial shapes changed during evaluation: '
                f'{self.spatial_shapes.tolist()} -> {shapes.tolist()}')

        level_ends = shapes.prod(dim=1).cumsum(dim=0).to(topk_indices.device)
        if topk_indices.numel() and topk_indices.max() >= level_ends[-1]:
            raise ValueError('enc_topk_indices contains an out-of-range flattened index')
        levels = torch.bucketize(topk_indices.contiguous(), level_ends, right=True)
        self.level_counts += torch.bincount(
            levels.flatten().cpu(), minlength=len(shapes))

        query_scores = topk_logits.sigmoid().max(dim=-1).values
        topk_xyxy = box_cxcywh_to_xyxy(topk_boxes.float())

        for batch_index, target in enumerate(targets):
            gt_xyxy = self._normalized_gt_xyxy(target, image_hw)
            areas = self._target_areas(target, gt_xyxy)
            num_queries = topk_xyxy.shape[1]
            self.query_count += num_queries

            if gt_xyxy.numel() == 0:
                query_best_iou = topk_xyxy.new_zeros(num_queries)
                gt_best_iou = topk_xyxy.new_zeros(0)
            else:
                ious, _ = box_iou(topk_xyxy[batch_index], gt_xyxy)
                query_best_iou = ious.max(dim=1).values
                gt_best_iou = ious.max(dim=0).values

            for threshold in self.IOU_THRESHOLDS:
                self.foreground[threshold] += int(
                    (query_best_iou >= threshold).sum().item())

            scores = query_scores[batch_index]
            background = query_best_iou < 0.1
            for threshold in self.SCORE_THRESHOLDS:
                self.high_conf_background[threshold] += int(
                    ((scores >= threshold) & background).sum().item())

            for scale_name, mask in self._scale_masks(areas).items():
                count = int(mask.sum().item())
                self.gt_counts[scale_name] += count
                if count == 0:
                    continue
                selected_best_iou = gt_best_iou[mask]
                self.best_iou_sums[scale_name] += selected_best_iou.sum().item()
                for threshold in self.IOU_THRESHOLDS:
                    self.recalled[threshold][scale_name] += int(
                        (selected_best_iou >= threshold).sum().item())

    @staticmethod
    def _ratio(numerator, denominator):
        return numerator / denominator if denominator else 0.0

    def summarize(self):
        print('\nEncoder Top-K Query Diagnosis\n')
        print('GT counts:')
        for name in self.SCALE_NAMES:
            print(f'{name}: {self.gt_counts[name]}')

        print('\nQuery Recall:')
        for threshold in self.IOU_THRESHOLDS:
            print(f'IoU >= {threshold:.1f}')
            for name in self.SCALE_NAMES:
                value = self._ratio(
                    self.recalled[threshold][name], self.gt_counts[name])
                print(f'{name}: {value:.6f}')

        print('\nMean Best Query IoU:')
        for name in self.SCALE_NAMES:
            value = self._ratio(self.best_iou_sums[name], self.gt_counts[name])
            print(f'{name}: {value:.6f}')

        print('\nSelected Query Foreground Ratio:')
        for threshold in self.IOU_THRESHOLDS:
            value = self._ratio(self.foreground[threshold], self.query_count)
            print(f'IoU >= {threshold:.1f}: {value:.6f}')

        print('\nHigh-confidence non-GT queries:')
        for threshold in self.SCORE_THRESHOLDS:
            value = self._ratio(
                self.high_conf_background[threshold], self.query_count)
            print(f'score >= {threshold:.1f} & IoU < 0.1: {value:.6f}')

        print('\nQuery source levels:')
        if self.level_counts is not None:
            for level, count_tensor in enumerate(self.level_counts):
                count = int(count_tensor.item())
                value = self._ratio(count, self.query_count)
                print(f'P{level + 3}: {count} ({value:.6f})')
