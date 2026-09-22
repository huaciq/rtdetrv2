"""Dataset-level diagnostics for RT-DETR encoder Top-K queries."""

import torch

from ..zoo.rtdetr.box_ops import box_cxcywh_to_xyxy, box_iou


class QueryStats:
    """Accumulate encoder-query coverage statistics for single-process eval."""

    IOU_THRESHOLDS = (0.1, 0.3, 0.5, 0.75, 0.9)
    TOP_N = (10, 20, 50, 100, 150, 300)
    TOP_N_IOU_THRESHOLDS = (0.3, 0.5, 0.75)
    CLASS_AWARE_IOU_THRESHOLDS = (0.3, 0.5, 0.75)
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
        self.topn_recalled = {
            n: {
                threshold: {name: 0 for name in self.SCALE_NAMES}
                for threshold in self.TOP_N_IOU_THRESHOLDS
            }
            for n in self.TOP_N
        }
        self.class_aware_recalled = {
            threshold: {name: 0 for name in self.SCALE_NAMES}
            for threshold in self.CLASS_AWARE_IOU_THRESHOLDS
        }
        self.best_ranks = {name: [] for name in self.SCALE_NAMES}
        self.topn_background = {n: 0 for n in self.TOP_N}
        self.topn_query_counts = {n: 0 for n in self.TOP_N}
        self.level_counts = None
        self.topn_level_counts = None
        self.best_level_counts = None
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
        if topk_boxes.shape[1] < self.TOP_N[-1]:
            raise ValueError(
                f'QueryStats requires at least Top-{self.TOP_N[-1]} queries, '
                f'got {topk_boxes.shape[1]}')

        shapes = spatial_shapes.detach().cpu().long()
        if self.spatial_shapes is None:
            self.spatial_shapes = shapes
            self.level_counts = torch.zeros(len(shapes), dtype=torch.long)
            self.topn_level_counts = {
                n: torch.zeros(len(shapes), dtype=torch.long)
                for n in self.TOP_N
            }
            self.best_level_counts = {
                name: torch.zeros(len(shapes), dtype=torch.long)
                for name in self.SCALE_NAMES
            }
        elif not torch.equal(self.spatial_shapes, shapes):
            raise ValueError(
                f'Encoder spatial shapes changed during evaluation: '
                f'{self.spatial_shapes.tolist()} -> {shapes.tolist()}')

        level_ends = shapes.prod(dim=1).cumsum(dim=0).to(topk_indices.device)
        if topk_indices.numel() and topk_indices.max() >= level_ends[-1]:
            raise ValueError('enc_topk_indices contains an out-of-range flattened index')
        levels = torch.bucketize(topk_indices.contiguous(), level_ends, right=True)
        levels = levels[:, :self.TOP_N[-1]]
        self.level_counts += torch.bincount(
            levels.flatten().cpu(), minlength=len(shapes))
        for n in self.TOP_N:
            self.topn_level_counts[n] += torch.bincount(
                levels[:, :n].flatten().cpu(), minlength=len(shapes))

        query_scores = topk_logits[:, :self.TOP_N[-1]].sigmoid().max(dim=-1).values
        query_classes = topk_logits[:, :self.TOP_N[-1]].argmax(dim=-1)
        topk_xyxy = box_cxcywh_to_xyxy(
            topk_boxes[:, :self.TOP_N[-1]].float())

        for batch_index, target in enumerate(targets):
            gt_xyxy = self._normalized_gt_xyxy(target, image_hw)
            areas = self._target_areas(target, gt_xyxy)
            labels = target['labels'].as_subclass(torch.Tensor).long()
            num_queries = topk_xyxy.shape[1]
            self.query_count += num_queries

            if labels.shape[0] != gt_xyxy.shape[0]:
                raise ValueError('Target labels and boxes must have the same length')
            if labels.numel() and (labels.min() < 0 or labels.max() >= topk_logits.shape[-1]):
                raise ValueError(
                    'Target labels are not valid encoder-logit indices: '
                    f'range [{labels.min().item()}, {labels.max().item()}], '
                    f'num_classes={topk_logits.shape[-1]}')

            if gt_xyxy.numel() == 0:
                query_best_iou = topk_xyxy.new_zeros(num_queries)
                gt_best_iou = topk_xyxy.new_zeros(0)
                gt_best_query_indices = torch.empty(
                    0, dtype=torch.long, device=topk_xyxy.device)
                ious = topk_xyxy.new_zeros((num_queries, 0))
            else:
                ious, _ = box_iou(topk_xyxy[batch_index], gt_xyxy)
                query_best_iou = ious.max(dim=1).values
                gt_best_iou, gt_best_query_indices = ious.max(dim=0)

            for threshold in self.IOU_THRESHOLDS:
                self.foreground[threshold] += int(
                    (query_best_iou >= threshold).sum().item())

            scores = query_scores[batch_index]
            background = query_best_iou < 0.1
            for threshold in self.SCORE_THRESHOLDS:
                self.high_conf_background[threshold] += int(
                    ((scores >= threshold) & background).sum().item())

            for n in self.TOP_N:
                self.topn_background[n] += int(
                    (query_best_iou[:n] < 0.1).sum().item())
                self.topn_query_counts[n] += n

            scale_masks = self._scale_masks(areas)
            for n in self.TOP_N:
                if gt_xyxy.numel() == 0:
                    topn_gt_best_iou = gt_best_iou
                else:
                    topn_gt_best_iou = ious[:n].max(dim=0).values
                for threshold in self.TOP_N_IOU_THRESHOLDS:
                    for scale_name, mask in scale_masks.items():
                        self.topn_recalled[n][threshold][scale_name] += int(
                            (topn_gt_best_iou[mask] >= threshold).sum().item())

            if gt_xyxy.numel():
                class_matches = (
                    query_classes[batch_index, :, None] == labels[None, :])
                for threshold in self.CLASS_AWARE_IOU_THRESHOLDS:
                    recalled = ((ious >= threshold) & class_matches).any(dim=0)
                    for scale_name, mask in scale_masks.items():
                        self.class_aware_recalled[threshold][scale_name] += int(
                            recalled[mask].sum().item())

                ranks = gt_best_query_indices + 1
                best_levels = levels[batch_index, gt_best_query_indices]
                for scale_name, mask in scale_masks.items():
                    self.best_ranks[scale_name].extend(
                        ranks[mask].detach().cpu().tolist())
                    self.best_level_counts[scale_name] += torch.bincount(
                        best_levels[mask].detach().cpu(), minlength=len(shapes))

            for scale_name, mask in scale_masks.items():
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

    @staticmethod
    def _rank_summary(ranks):
        if not ranks:
            return 0.0, 0.0, 0.0, 0.0
        values = torch.tensor(ranks, dtype=torch.float64)
        quantiles = torch.quantile(values, torch.tensor(
            [0.5, 0.75, 0.9], dtype=torch.float64))
        return (values.mean().item(), quantiles[0].item(),
                quantiles[1].item(), quantiles[2].item())

    def summarize(self):
        print('\nEncoder Top-K Query Diagnosis\n')
        print('GT counts:')
        for name in self.SCALE_NAMES:
            print(f'{name}: {self.gt_counts[name]}')

        print('\nQuery Recall:')
        for threshold in self.IOU_THRESHOLDS:
            print(f'IoU >= {threshold:g}')
            for name in self.SCALE_NAMES:
                value = self._ratio(
                    self.recalled[threshold][name], self.gt_counts[name])
                print(f'{name}: {value:.6f}')

        print('\nTop-N Query Recall Curve:')
        for n in self.TOP_N:
            print(f'Top{n}:')
            for threshold in self.TOP_N_IOU_THRESHOLDS:
                print(f'  QR@IoU={threshold:.2f}')
                for name in self.SCALE_NAMES:
                    value = self._ratio(
                        self.topn_recalled[n][threshold][name],
                        self.gt_counts[name])
                    print(f'  {name}: {value:.6f}')

        print('\nBest Query Rank:')
        for name in self.SCALE_NAMES:
            mean, median, p75, p90 = self._rank_summary(
                self.best_ranks[name])
            print(f'{name}: mean={mean:.3f}, median={median:.3f}, '
                  f'P75={p75:.3f}, P90={p90:.3f}')

        print('\nClass-aware Query Recall:')
        for threshold in self.CLASS_AWARE_IOU_THRESHOLDS:
            print(f'IoU >= {threshold:.2f}')
            for name in self.SCALE_NAMES:
                value = self._ratio(
                    self.class_aware_recalled[threshold][name],
                    self.gt_counts[name])
                print(f'{name}: {value:.6f}')

        print('\nMean Best Query IoU:')
        for name in self.SCALE_NAMES:
            value = self._ratio(self.best_iou_sums[name], self.gt_counts[name])
            print(f'{name}: {value:.6f}')

        print('\nSelected Query Foreground Ratio:')
        for threshold in self.IOU_THRESHOLDS:
            value = self._ratio(self.foreground[threshold], self.query_count)
            print(f'IoU >= {threshold:g}: {value:.6f}')

        print('\nHigh-confidence non-GT queries:')
        for threshold in self.SCORE_THRESHOLDS:
            value = self._ratio(
                self.high_conf_background[threshold], self.query_count)
            print(f'score >= {threshold:.1f} & IoU < 0.1: {value:.6f}')

        print('\nTop-N Background Occupancy (max IoU < 0.1):')
        for n in self.TOP_N:
            count = self.topn_background[n]
            value = self._ratio(count, self.topn_query_counts[n])
            print(f'Top{n}: {count} ({value:.6f})')

        print('\nQuery source levels:')
        if self.level_counts is not None:
            for level, count_tensor in enumerate(self.level_counts):
                count = int(count_tensor.item())
                value = self._ratio(count, self.query_count)
                print(f'P{level + 3}: {count} ({value:.6f})')

        print('\nTop-N Feature Level Distribution:')
        if self.topn_level_counts is not None:
            for n in self.TOP_N:
                print(f'Top{n}:')
                total = int(self.topn_level_counts[n].sum().item())
                for level, count_tensor in enumerate(self.topn_level_counts[n]):
                    count = int(count_tensor.item())
                    value = self._ratio(count, total)
                    print(f'  P{level + 3}: {count} ({value:.6f})')

        print('\nBest-matching Query Level by GT Size:')
        if self.best_level_counts is not None:
            for name in ('small', 'medium', 'large'):
                print(f'{name}:')
                total = int(self.best_level_counts[name].sum().item())
                for level, count_tensor in enumerate(self.best_level_counts[name]):
                    count = int(count_tensor.item())
                    value = self._ratio(count, total)
                    print(f'  P{level + 3}: {count} ({value:.6f})')
