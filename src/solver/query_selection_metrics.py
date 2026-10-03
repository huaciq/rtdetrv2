"""Compact encoder query metrics, deduplicated by image ID across DDP ranks."""

import torch

from ..misc import dist_utils
from ..zoo.rtdetr.box_ops import box_cxcywh_to_xyxy, box_iou
from .query_stats import QueryStats


class QuerySelectionMetrics:
    """Use the same small-GT/geometry definitions as existing QueryStats.

    Rank is one-based within the actual selected encoder Top-K, in its
    selection-score order. For every small GT, take the query with maximum
    IoU (first rank wins ties). This is class-agnostic geometric coverage,
    not decoder detection recall or one-to-one matching.
    """

    def __init__(self):
        self.images = {}

    @torch.no_grad()
    def update(self, outputs, targets, image_hw):
        boxes = box_cxcywh_to_xyxy(outputs['enc_topk_boxes'].float())
        if boxes.ndim != 3 or boxes.shape[1] == 0 or boxes.shape[0] != len(targets):
            raise ValueError('Expected nonempty encoder Top-K boxes [B, K, 4]')
        for proposals, target in zip(boxes, targets):
            image_id = int(target['image_id'].item())
            if image_id in self.images:
                continue
            gt = QueryStats._normalized_gt_xyxy(target, image_hw)
            small = QueryStats._target_areas(target, gt) < 32 ** 2
            count = int(small.sum().item())
            if count:
                ious = box_iou(proposals, gt[small])[0]
                best_iou, best_query = ious.max(dim=0)
                ranks = (best_query + 1).cpu().tolist()
                recalled = int((best_iou >= 0.75).sum().item())
            else:
                ranks, recalled = [], 0
            self.images[image_id] = {
                'small_gt_count': count, 'small_recalled': recalled,
                'small_best_query_ranks': ranks, 'num_queries': len(proposals),
            }

    @staticmethod
    def merge_image_records(shards):
        # DistributedSampler may pad the validation split with repeated IDs.
        # Keep one observation per image, matching COCO's unique-image protocol.
        images = {}
        for shard in shards:
            for image_id, record in shard.items():
                images.setdefault(image_id, record)
        return images

    def summarize(self):
        images = self.merge_image_records(dist_utils.all_gather(self.images))
        count = sum(record['small_gt_count'] for record in images.values())
        recalled = sum(record['small_recalled'] for record in images.values())
        ranks = torch.tensor([
            rank for record in images.values()
            for rank in record['small_best_query_ranks']], dtype=torch.float64)
        rank_stats = {'mean': None, 'median': None, 'P90': None}
        if ranks.numel():
            rank_stats = {
                'mean': float(ranks.mean()),
                'median': float(torch.quantile(ranks, 0.5)),
                'P90': float(torch.quantile(ranks, 0.9)),
            }
        return {
            'small_query_recall_iou75': recalled / count if count else None,
            'small_best_query_rank': rank_stats,
            'small_gt_count': count, 'small_recalled_count': recalled,
            'unique_image_count': len(images),
            'num_queries': sorted({r['num_queries'] for r in images.values()}),
            'protocol': {
                'stage': 'selected encoder proposals before decoder refinement',
                'small': 'annotation area < 32^2 in original image pixels',
                'recall': 'class-agnostic max IoU over all selected queries >= 0.75',
                'rank': '1-based selection-score order; max-IoU query; first tie',
                'rank_population': 'all small GTs, including unrecalled and zero-IoU GTs',
                'quantiles': 'linear interpolation',
                'distributed': 'deduplicated by image_id',
            },
        }
