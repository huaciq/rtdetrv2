"""
Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
https://github.com/facebookresearch/detr/blob/main/engine.py

Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

import sys
import math
from typing import Iterable

import torch
import torch.amp 
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp.grad_scaler import GradScaler

from ..optim import ModelEMA, Warmup
from ..data import CocoEvaluator
from ..misc import MetricLogger, SmoothedValue, dist_utils
from .query_stats import QueryStats


def optimized_loss(loss_dict):
    """Exclude detached metric-only entries from the optimized objective."""
    return sum(value for key, value in loss_dict.items() if not key.endswith('_raw'))


def _mean_rank_gradient_norm(loss, parameters):
    """Return the mean per-rank L2 norm without modifying parameter ``.grad``."""
    gradients = torch.autograd.grad(
        loss,
        parameters,
        retain_graph=True,
        allow_unused=True,
    )
    squared_norm = torch.zeros((), device=loss.device, dtype=torch.float32)
    for gradient in gradients:
        if gradient is not None:
            squared_norm += gradient.detach().float().pow(2).sum()
    norm = squared_norm.sqrt()
    if dist_utils.is_dist_available_and_initialized():
        torch.distributed.all_reduce(norm, op=torch.distributed.ReduceOp.SUM)
        norm /= dist_utils.get_world_size()
    return norm


def measure_gkd_gradient_norms(model, criterion, loss_dict, global_step):
    """Measure detection and weighted-GKD gradient norms at sampled steps."""
    interval = int(getattr(criterion, 'grad_norm_interval', 0))
    if interval <= 0 or global_step % interval != 0 or 'loss_gkd' not in loss_dict:
        return None

    parameters = tuple(parameter for parameter in model.parameters() if parameter.requires_grad)
    detection_loss = sum(
        value for key, value in loss_dict.items()
        if not key.endswith('_raw') and key != 'loss_gkd'
    )
    gkd_loss = loss_dict['loss_gkd']
    grad_norm_det = _mean_rank_gradient_norm(detection_loss, parameters)
    grad_norm_gkd = _mean_rank_gradient_norm(gkd_loss, parameters)
    grad_ratio = grad_norm_gkd / grad_norm_det.clamp_min(1e-12)
    return {
        'grad_norm_det': grad_norm_det.detach(),
        'grad_norm_gkd': grad_norm_gkd.detach(),
        'grad_ratio': grad_ratio.detach(),
    }


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0, **kwargs):
    model.train()
    criterion.train()
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    
    print_freq = kwargs.get('print_freq', 10)
    writer :SummaryWriter = kwargs.get('writer', None)

    ema :ModelEMA = kwargs.get('ema', None)
    scaler :GradScaler = kwargs.get('scaler', None)
    lr_warmup_scheduler :Warmup = kwargs.get('lr_warmup_scheduler', None)

    for i, (samples, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        global_step = epoch * len(data_loader) + i
        metas = dict(epoch=epoch, step=i, global_step=global_step)

        if scaler is not None:
            with torch.autocast(device_type=str(device), cache_enabled=True):
                outputs = model(samples, targets=targets)
            
            with torch.autocast(device_type=str(device), enabled=False):
                loss_dict = criterion(outputs, targets, **metas)

            loss = optimized_loss(loss_dict)
            grad_metrics = measure_gkd_gradient_norms(
                model, criterion, loss_dict, global_step
            )
            scaler.scale(loss).backward()
            
            if max_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        else:
            outputs = model(samples, targets=targets)
            loss_dict = criterion(outputs, targets, **metas)

            loss : torch.Tensor = optimized_loss(loss_dict)
            grad_metrics = measure_gkd_gradient_norms(
                model, criterion, loss_dict, global_step
            )
            optimizer.zero_grad()
            loss.backward()
            
            if max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

            optimizer.step()
        
        # ema 
        if ema is not None:
            ema.update(model)

        if lr_warmup_scheduler is not None:
            lr_warmup_scheduler.step()

        loss_dict_reduced = dist_utils.reduce_dict(loss_dict)
        loss_value = optimized_loss(loss_dict_reduced)

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        metric_logger.update(loss=loss_value, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        if grad_metrics is not None:
            metric_logger.update(**grad_metrics)
            if dist_utils.is_main_process():
                print(
                    "GKD_GRAD_NORM "
                    f"global_step={global_step} "
                    f"grad_norm_det={grad_metrics['grad_norm_det'].item():.8f} "
                    f"grad_norm_gkd={grad_metrics['grad_norm_gkd'].item():.8f} "
                    f"grad_ratio={grad_metrics['grad_ratio'].item():.8f}",
                    flush=True,
                )

        if writer and dist_utils.is_main_process():
            writer.add_scalar('Loss/total', loss_value.item(), global_step)
            for j, pg in enumerate(optimizer.param_groups):
                writer.add_scalar(f'Lr/pg_{j}', pg['lr'], global_step)
            for k, v in loss_dict_reduced.items():
                writer.add_scalar(f'Loss/{k}', v.item(), global_step)
            if grad_metrics is not None:
                for key, value in grad_metrics.items():
                    writer.add_scalar(f'Gradient/{key}', value.item(), global_step)
                
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(model: torch.nn.Module, criterion: torch.nn.Module, postprocessor, data_loader, coco_evaluator: CocoEvaluator, device):
    model.eval()
    criterion.eval()
    coco_evaluator.cleanup()
    iou_types = coco_evaluator.iou_types

    metric_logger = MetricLogger(delimiter="  ")
    header = 'Test:'
    query_stats = None
    query_stats_skipped = False
    
    for samples, targets in metric_logger.log_every(data_loader, 10, header):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        outputs = model(samples)

        if 'enc_topk_boxes' in outputs:
            if dist_utils.get_world_size() == 1:
                if query_stats is None:
                    query_stats = QueryStats()
                query_stats.update(outputs, targets, samples.shape[-2:])
            elif not query_stats_skipped:
                print('Encoder Top-K Query Diagnosis skipped: '
                      'the first implementation supports single-GPU evaluation only.')
                query_stats_skipped = True

        # TODO (lyuwenyu), fix dataset converted using `convert_to_coco_api`?
        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        
        results = postprocessor(outputs, orig_target_sizes)

        # if 'segm' in postprocessor.keys():
        #     target_sizes = torch.stack([t["size"] for t in targets], dim=0)
        #     results = postprocessor['segm'](results, outputs, orig_target_sizes, target_sizes)

        res = {target['image_id'].item(): output for target, output in zip(targets, results)}
        if coco_evaluator is not None:
            coco_evaluator.update(res)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    if coco_evaluator is not None:
        coco_evaluator.accumulate()
        coco_evaluator.summarize()

    if query_stats is not None:
        query_stats.summarize()

    stats = {}
    # stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    if coco_evaluator is not None:
        if 'bbox' in iou_types:
            stats['coco_eval_bbox'] = coco_evaluator.coco_eval['bbox'].stats.tolist()
        if 'segm' in iou_types:
            stats['coco_eval_masks'] = coco_evaluator.coco_eval['segm'].stats.tolist()
            
    return stats, coco_evaluator

