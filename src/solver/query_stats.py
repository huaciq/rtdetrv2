"""Dataset-level diagnostics for RT-DETR encoder Top-K queries."""

import csv
import json
import random
from pathlib import Path

import torch
from PIL import Image, ImageDraw

from ..zoo.rtdetr.box_ops import box_cxcywh_to_xyxy, box_iou


class QueryStats:
    """Accumulate encoder-query coverage statistics for single-process eval."""

    IOU_THRESHOLDS = (0.1, 0.3, 0.5, 0.75, 0.9)
    TOP_N = (10, 20, 50, 100, 150, 300)
    TOP_N_IOU_THRESHOLDS = (0.3, 0.5, 0.75)
    CLASS_AWARE_IOU_THRESHOLDS = (0.3, 0.5, 0.75)
    SCORE_THRESHOLDS = (0.1, 0.3, 0.5)
    SCALE_NAMES = ('all', 'small', 'medium', 'large')
    BACKGROUND_IOU_THRESHOLD = 0.1
    VISUALIZATION_LIMIT = 50

    def __init__(self, output_dir=None, visualization_seed=0):
        self.output_dir = Path(output_dir) if output_dir is not None else None
        self.visualization_rng = random.Random(visualization_seed)
        self.visualization_candidate_count = 0
        self.visualization_samples = []
        self.background_metadata = []
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
        self.correlation_chunks = {
            name: {'scores': [], 'ious': []}
            for name in self.SCALE_NAMES
        }
        self.gt_query_comparison = {
            name: {
                'best_iou': {
                    'count': 0, 'iou_sum': 0.0,
                    'score_sum': 0.0, 'rank_sum': 0.0,
                    'level_counts': None,
                },
                'best_score': {
                    'count': 0, 'iou_sum': 0.0,
                    'score_sum': 0.0, 'rank_sum': 0.0,
                    'level_counts': None,
                },
                'same_query_count': 0,
            }
            for name in self.SCALE_NAMES
        }
        self.level_counts = None
        self.topn_level_counts = None
        self.best_level_counts = None
        self.small_best_by_level = None
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

    @staticmethod
    def _accumulate_choice(bucket, mask, ious, scores, ranks, levels,
                           num_levels):
        count = int(mask.sum().item())
        if count == 0:
            return
        bucket['count'] += count
        bucket['iou_sum'] += ious[mask].sum().item()
        bucket['score_sum'] += scores[mask].sum().item()
        bucket['rank_sum'] += ranks[mask].float().sum().item()
        bucket['level_counts'] += torch.bincount(
            levels[mask].detach().cpu(), minlength=num_levels)

    def _collect_background_artifacts(
            self, batch_index, target, query_boxes, query_scores,
            query_classes, query_levels, query_best_iou, gt_xyxy,
            image_hw, diagnostic_images):
        background_indices = torch.nonzero(
            query_best_iou[:10] < self.BACKGROUND_IOU_THRESHOLD,
            as_tuple=False).flatten()
        if background_indices.numel() == 0:
            return

        image_h, image_w = image_hw
        image_id = int(target['image_id'].item())
        pixel_scale = query_boxes.new_tensor(
            [image_w, image_h, image_w, image_h])
        image_records = []
        for query_index in background_indices.tolist():
            normalized_box = query_boxes[query_index].detach().float().cpu()
            pixel_box = (query_boxes[query_index] * pixel_scale).detach().float().cpu()
            record = {
                'image_id': image_id,
                'rank': query_index + 1,
                'score': float(query_scores[query_index].item()),
                'predicted_class': int(query_classes[query_index].item()),
                'feature_level': f'P{int(query_levels[query_index].item()) + 3}',
                'max_iou': float(query_best_iou[query_index].item()),
                'box_xyxy_normalized': normalized_box.tolist(),
                'box_xyxy_input_pixels': pixel_box.tolist(),
            }
            self.background_metadata.append(record)
            image_records.append(record)

        if diagnostic_images is None:
            return

        self.visualization_candidate_count += 1
        replacement = None
        if len(self.visualization_samples) >= self.VISUALIZATION_LIMIT:
            replacement = self.visualization_rng.randrange(
                self.visualization_candidate_count)
            if replacement >= self.VISUALIZATION_LIMIT:
                return

        visualization = {
            'image_id': image_id,
            'image': (diagnostic_images[batch_index].detach()
                      .float().clamp(0, 1).mul(255).byte().cpu()),
            'gt_boxes': (gt_xyxy * pixel_scale).detach().float().cpu(),
            'queries': image_records,
        }
        if replacement is None:
            self.visualization_samples.append(visualization)
        else:
            self.visualization_samples[replacement] = visualization

    @staticmethod
    def _render_visualization(sample, output_path):
        image_tensor = sample['image']
        if image_tensor.shape[0] == 1:
            image_tensor = image_tensor.repeat(3, 1, 1)
        image_array = image_tensor[:3].permute(1, 2, 0).numpy()
        image = Image.fromarray(image_array)
        draw = ImageDraw.Draw(image)

        for box in sample['gt_boxes'].tolist():
            draw.rectangle(box, outline=(0, 255, 0), width=2)

        for query in sample['queries']:
            box = query['box_xyxy_input_pixels']
            draw.rectangle(box, outline=(255, 0, 0), width=2)
            label = (f"r{query['rank']} {query['score']:.3f} "
                     f"{query['feature_level']}")
            text_x = max(0, box[0])
            text_y = max(0, box[1] - 12)
            draw.text((text_x, text_y), label, fill=(255, 0, 0))

        image.save(output_path)

    def _write_background_artifacts(self):
        if self.output_dir is None:
            return

        self.output_dir.mkdir(parents=True, exist_ok=True)
        json_path = self.output_dir / 'top10_background_queries.json'
        csv_path = self.output_dir / 'top10_background_queries.csv'
        with json_path.open('w', encoding='utf-8') as file:
            json.dump(self.background_metadata, file, indent=2)

        csv_fields = (
            'image_id', 'rank', 'score', 'predicted_class',
            'feature_level', 'max_iou',
            'normalized_x1', 'normalized_y1',
            'normalized_x2', 'normalized_y2',
            'input_x1', 'input_y1', 'input_x2', 'input_y2',
        )
        with csv_path.open('w', encoding='utf-8', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=csv_fields)
            writer.writeheader()
            for record in self.background_metadata:
                normalized = record['box_xyxy_normalized']
                pixels = record['box_xyxy_input_pixels']
                writer.writerow({
                    'image_id': record['image_id'],
                    'rank': record['rank'],
                    'score': record['score'],
                    'predicted_class': record['predicted_class'],
                    'feature_level': record['feature_level'],
                    'max_iou': record['max_iou'],
                    'normalized_x1': normalized[0],
                    'normalized_y1': normalized[1],
                    'normalized_x2': normalized[2],
                    'normalized_y2': normalized[3],
                    'input_x1': pixels[0],
                    'input_y1': pixels[1],
                    'input_x2': pixels[2],
                    'input_y2': pixels[3],
                })

        visualization_dir = self.output_dir / 'top10_background_visualizations'
        visualization_dir.mkdir(parents=True, exist_ok=True)
        for sample in self.visualization_samples:
            output_path = visualization_dir / f"image_{sample['image_id']}.png"
            self._render_visualization(sample, output_path)

        print('\nTop-10 background artifacts:')
        print(f'metadata queries: {len(self.background_metadata)}')
        print(f'candidate images: {self.visualization_candidate_count}')
        print(f'visualized images: {len(self.visualization_samples)}')
        print(f'JSON: {json_path}')
        print(f'CSV: {csv_path}')
        print(f'visualizations: {visualization_dir}')

    def update(self, outputs, targets, image_hw, diagnostic_images=None):
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
        if diagnostic_images is not None and diagnostic_images.shape[0] != len(targets):
            raise ValueError('Diagnostic image batch size does not match targets')
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
            self.small_best_by_level = {
                level: {'ranks': [], 'iou_sum': 0.0, 'score_sum': 0.0}
                for level in range(len(shapes))
            }
            for comparison in self.gt_query_comparison.values():
                comparison['best_iou']['level_counts'] = torch.zeros(
                    len(shapes), dtype=torch.long)
                comparison['best_score']['level_counts'] = torch.zeros(
                    len(shapes), dtype=torch.long)
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
            scale_masks = self._scale_masks(areas)
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
                query_best_gt_indices = torch.empty(
                    num_queries, dtype=torch.long, device=topk_xyxy.device)
                ious = topk_xyxy.new_zeros((num_queries, 0))
            else:
                ious, _ = box_iou(topk_xyxy[batch_index], gt_xyxy)
                query_best_iou, query_best_gt_indices = ious.max(dim=1)
                gt_best_iou, gt_best_query_indices = ious.max(dim=0)

            image_scores = query_scores[batch_index]
            self.correlation_chunks['all']['scores'].append(
                image_scores.detach().float().cpu())
            self.correlation_chunks['all']['ious'].append(
                query_best_iou.detach().float().cpu())
            if gt_xyxy.numel():
                positive_overlap = query_best_iou > 0
                for scale_name in ('small', 'medium', 'large'):
                    matched_scale = (
                        scale_masks[scale_name][query_best_gt_indices]
                        & positive_overlap)
                    if matched_scale.any():
                        self.correlation_chunks[scale_name]['scores'].append(
                            image_scores[matched_scale].detach().float().cpu())
                        self.correlation_chunks[scale_name]['ious'].append(
                            query_best_iou[matched_scale].detach().float().cpu())

            for threshold in self.IOU_THRESHOLDS:
                self.foreground[threshold] += int(
                    (query_best_iou >= threshold).sum().item())

            scores = image_scores
            background = query_best_iou < 0.1
            for threshold in self.SCORE_THRESHOLDS:
                self.high_conf_background[threshold] += int(
                    ((scores >= threshold) & background).sum().item())

            for n in self.TOP_N:
                self.topn_background[n] += int(
                    (query_best_iou[:n] < 0.1).sum().item())
                self.topn_query_counts[n] += n

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
                best_scores = scores[gt_best_query_indices]

                overlap = ious > self.BACKGROUND_IOU_THRESHOLD
                overlap_scores = scores[:, None].masked_fill(~overlap, -torch.inf)
                best_overlap_scores, best_score_indices = overlap_scores.max(dim=0)
                best_score_valid = overlap.any(dim=0)
                best_score_ious = ious.gather(
                    0, best_score_indices.unsqueeze(0)).squeeze(0)
                best_score_ranks = best_score_indices + 1
                best_score_levels = levels[batch_index, best_score_indices]

                for scale_name, mask in scale_masks.items():
                    self.best_ranks[scale_name].extend(
                        ranks[mask].detach().cpu().tolist())
                    self.best_level_counts[scale_name] += torch.bincount(
                        best_levels[mask].detach().cpu(), minlength=len(shapes))

                    comparison = self.gt_query_comparison[scale_name]
                    self._accumulate_choice(
                        comparison['best_iou'], mask, gt_best_iou,
                        best_scores, ranks, best_levels, len(shapes))
                    valid_mask = mask & best_score_valid
                    self._accumulate_choice(
                        comparison['best_score'], valid_mask,
                        best_score_ious, best_overlap_scores,
                        best_score_ranks, best_score_levels, len(shapes))
                    comparison['same_query_count'] += int(
                        ((best_score_indices == gt_best_query_indices)
                         & valid_mask).sum().item())

                small_mask = scale_masks['small']
                for level in range(len(shapes)):
                    level_mask = small_mask & (best_levels == level)
                    count = int(level_mask.sum().item())
                    if count == 0:
                        continue
                    group = self.small_best_by_level[level]
                    group['ranks'].extend(
                        ranks[level_mask].detach().cpu().tolist())
                    group['iou_sum'] += gt_best_iou[level_mask].sum().item()
                    group['score_sum'] += best_scores[level_mask].sum().item()

            if self.output_dir is not None:
                self._collect_background_artifacts(
                    batch_index, target, topk_xyxy[batch_index], scores,
                    query_classes[batch_index], levels[batch_index],
                    query_best_iou, gt_xyxy, image_hw, diagnostic_images)

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

    @staticmethod
    def _average_ranks(values):
        sorted_values, order = values.sort()
        _, counts = torch.unique_consecutive(
            sorted_values, return_counts=True)
        starts = counts.cumsum(0) - counts
        average = starts.to(torch.float64) + (
            counts.to(torch.float64) - 1) / 2
        sorted_ranks = torch.repeat_interleave(average, counts)
        ranks = torch.empty_like(sorted_ranks)
        ranks[order] = sorted_ranks
        return ranks

    @staticmethod
    def _pearson(x, y):
        if x.numel() < 2:
            return float('nan')
        x = x.to(torch.float64)
        y = y.to(torch.float64)
        x = x - x.mean()
        y = y - y.mean()
        denominator = x.square().sum().sqrt() * y.square().sum().sqrt()
        if denominator == 0:
            return float('nan')
        return (x * y).sum().div(denominator).item()

    @classmethod
    def _correlations(cls, scores, ious):
        if not scores:
            return 0, float('nan'), float('nan')
        scores = torch.cat(scores)
        ious = torch.cat(ious)
        pearson = cls._pearson(scores, ious)
        spearman = cls._pearson(
            cls._average_ranks(scores), cls._average_ranks(ious))
        return scores.numel(), pearson, spearman

    @staticmethod
    def _format_metric(value):
        return 'N/A' if value != value else f'{value:.6f}'

    def _print_choice_summary(self, label, bucket):
        count = bucket['count']
        print(f'  {label} (n={count}):')
        print(f"    mean IoU: {self._ratio(bucket['iou_sum'], count):.6f}")
        print(f"    mean score: {self._ratio(bucket['score_sum'], count):.6f}")
        print(f"    mean rank: {self._ratio(bucket['rank_sum'], count):.3f}")
        print('    feature levels:')
        for level, count_tensor in enumerate(bucket['level_counts']):
            level_count = int(count_tensor.item())
            print(f'      P{level + 3}: {level_count} '
                  f'({self._ratio(level_count, count):.6f})')

    def summarize(self):
        print('\nEncoder Top-K Query Diagnosis\n')
        print('GT counts:')
        for name in self.SCALE_NAMES:
            print(f'{name}: {self.gt_counts[name]}')

        print('\nScore-IoU Correlation:')
        for name in self.SCALE_NAMES:
            chunks = self.correlation_chunks[name]
            count, pearson, spearman = self._correlations(
                chunks['scores'], chunks['ious'])
            suffix = '' if name == 'all' else ' (best-matched, IoU > 0)'
            print(f'{name}{suffix} (n={count}):')
            print(f'  Pearson: {self._format_metric(pearson)}')
            print(f'  Spearman: {self._format_metric(spearman)}')

        print('\nGT Best-IoU Query vs Highest-Score Overlapping Query:')
        for name in ('small', 'medium', 'large'):
            print(f'{name}:')
            comparison = self.gt_query_comparison[name]
            self._print_choice_summary('q_best_iou', comparison['best_iou'])
            self._print_choice_summary(
                'q_best_score (IoU > 0.1)', comparison['best_score'])
            valid_count = comparison['best_score']['count']
            if valid_count:
                same_ratio = self._ratio(
                    comparison['same_query_count'], valid_count)
                print(f"  same query: {comparison['same_query_count']} "
                      f'({same_ratio:.6f}); different: {1 - same_ratio:.6f}')
            else:
                print('  same query: N/A (no query with IoU > 0.1)')

        print('\nSmall GT Best Query Rank by Feature Level:')
        if self.small_best_by_level is not None:
            for level, group in self.small_best_by_level.items():
                mean, median, p75, p90 = self._rank_summary(group['ranks'])
                count = len(group['ranks'])
                print(f'P{level + 3} (n={count}): mean rank={mean:.3f}, '
                      f'median={median:.3f}, P75={p75:.3f}, P90={p90:.3f}, '
                      f"mean IoU={self._ratio(group['iou_sum'], count):.6f}, "
                      f"mean score={self._ratio(group['score_sum'], count):.6f}")

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

        self._write_background_artifacts()
