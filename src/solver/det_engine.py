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
from .query_stats import (
    ClassConditionedQualityProbeStats,
    FinalQualityProbeStats,
    MultiThresholdClassConditionedQualityProbeStats,
    QueryStats,
)
from .query_selection_metrics import QuerySelectionMetrics
from ..zoo.rtdetr.umqr import box_keypoints
from ..zoo.rtdetr.box_ops import box_cxcywh_to_xyxy


def optimized_loss(loss_dict):
    """Exclude detached metric-only entries from the optimized objective."""
    return sum(value for key, value in loss_dict.items() if not key.endswith('_raw'))


def check_umqr_first_batch(model, criterion, outputs, targets, loss_dict, global_step):
    """One correctness report after backward, before gradients are cleared."""
    if not getattr(criterion, 'umqr', False) or getattr(criterion, '_umqr_debug_done', False):
        return
    points = outputs['pred_keypoints'].detach()
    scales = outputs['pred_keypoint_log_scales'].detach()
    gt = box_keypoints(torch.cat([target['boxes'].float() for target in targets]))
    parameters = list(dist_utils.de_parallel(model).decoder.umqr_head.named_parameters())
    with_grad = sum(parameter.grad is not None for _, parameter in parameters)
    gradients_finite = all(parameter.grad is not None and torch.isfinite(parameter.grad).all().item()
                           for _, parameter in parameters)
    values_finite = all(torch.isfinite(tensor).all().item() for tensor in (points, scales, gt))
    values_finite &= all(torch.isfinite(value).all().item() for key, value in loss_dict.items()
                         if key.startswith('loss_ukp'))
    gt_valid = not gt.numel() or ((gt >= -1e-5) & (gt <= 1 + 1e-5)).all().item()
    shape_valid = points.shape == (*outputs['pred_boxes'].shape[:2], 5, 2) and scales.shape == points.shape
    direct_details = {}
    head = dist_utils.de_parallel(model).decoder.umqr_head
    if head.direct_box:
        boxes = outputs['pred_keypoint_boxes'].detach()
        confidence = outputs['pred_geometry_confidence'].detach()
        corners = box_cxcywh_to_xyxy(boxes)
        geometry_valid = ((boxes >= 0) & (boxes <= 1)).all().item()
        geometry_valid &= (boxes[..., 2:] > 0).all().item()
        geometry_valid &= ((corners >= -1e-6) & (corners <= 1 + 1e-6)).all().item()
        geometry_valid &= ((confidence > 0) & (confidence < 1)).all().item()
        geometry_valid &= boxes.shape == outputs['pred_boxes'].shape
        geometry_valid &= confidence.shape == (*points.shape[:2], 1)
        if global_step == 0:
            geometry_valid &= float(head.alpha.detach()) == 0.
        values_finite &= all(torch.isfinite(tensor).all().item()
                             for tensor in (boxes, confidence, head.alpha, outputs['pred_boxes']))
        shape_valid &= geometry_valid
        direct_details = {
            'B_kp_legal': geometry_valid,
            'B_kp_range': [float(boxes.min()), float(boxes.max())],
            'width_height_range': [float(boxes[..., 2:].min()), float(boxes[..., 2:].max())],
            'C_geo_range': [float(confidence.min()), float(confidence.max())],
            'alpha': float(head.alpha.detach()),
            'alpha_has_gradient': head.alpha.grad is not None,
            'alpha_gradient': float(head.alpha.grad.detach()) if head.alpha.grad is not None else None,
        }
    local_ok = bool(shape_valid and values_finite and gradients_finite and gt_valid)
    all_ok = torch.tensor(int(local_ok), device=points.device)
    if dist_utils.is_dist_available_and_initialized():
        torch.distributed.all_reduce(all_ok, op=torch.distributed.ReduceOp.MIN)
    if dist_utils.is_main_process():
        print('UMQR_FIRST_BATCH', {
            'global_step': global_step, 'keypoints_shape': list(points.shape),
            'GT_keypoint_range': [float(gt.min()), float(gt.max())] if gt.numel() else None,
            'log_scale_range': [float(scales.min()), float(scales.max())],
            'sigma_range': [float(scales.min().exp()), float(scales.max().exp())],
            'L_ukp': float(loss_dict['loss_ukp'].detach()),
            'aux_L_ukp': {key: float(value.detach()) for key, value in loss_dict.items()
                          if key.startswith('loss_ukp_aux_')},
            'parameters_with_gradient': f'{with_grad}/{len(parameters)}',
            'gradients_finite': gradients_finite, 'values_finite': values_finite,
            'all_ranks_passed': bool(all_ok.item()),
            **direct_details,
        }, flush=True)
    if not all_ok.item():
        raise FloatingPointError('UMQR first-batch shape/GT/finite/gradient check failed on a rank')
    criterion._umqr_debug_done = True


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


def check_sber_first_batch(model, outputs, loss_dict, global_step):
    """One all-rank correctness check after ordinary detection backward."""
    decoder = dist_utils.de_parallel(model).decoder
    if not getattr(decoder, 'sber', False):
        return
    head = decoder.sber_head
    if getattr(head, '_debug_done', False):
        return
    details = outputs.get('sber_debug', [])
    boxes = outputs['pred_boxes'].detach()
    corners = box_cxcywh_to_xyxy(boxes)
    box_tensors = [boxes]
    for key in ('aux_outputs', 'dn_aux_outputs'):
        box_tensors.extend(output['pred_boxes'].detach() for output in outputs.get(key, []))
    scale_aware = getattr(decoder, 'sber_scale_aware', False)
    boxes_legal = all(
        ((box >= 0) & (box <= 1)).all().item() and (box[..., 2:] > 0).all().item()
        and (scale_aware or ((box_cxcywh_to_xyxy(box) >= -1e-6)
                            & (box_cxcywh_to_xyxy(box) <= 1 + 1e-6)).all().item())
        for box in box_tensors)
    parameters = list(head.named_parameters())
    gradients_finite = all(parameter.grad is not None and torch.isfinite(parameter.grad).all().item()
                           for _, parameter in parameters)
    values_finite = all(torch.isfinite(value).all().item()
                        for value in (*box_tensors, *loss_dict.values()))
    detail_valid = bool(details)
    reports = []
    for detail in details:
        values_finite &= all(torch.isfinite(value).all().item() for value in detail.values()
                             if isinstance(value, torch.Tensor))
        detail_valid &= bool(detail['evidence_finite'].item())
        detail_valid &= 0 <= float(detail['sampling_min']) <= float(detail['sampling_max']) <= 1
        detail_valid &= 0 < float(detail['gate_min']) <= float(detail['gate_max']) < 1
        detail_valid &= float(detail['offset_fraction_abs_max']) <= head.rho + 1e-6
        if scale_aware:
            # Baseline cxcywh is normalized, but its image corners may extend
            # outside [0,1]. Only enabled boxes undergo E4's legal projection.
            detail_valid &= bool(detail['enabled_box_corners_legal'].item())
            detail_valid &= detail['large_boundary_residual_abs_max'].item() == 0.
            detail_valid &= detail['large_offset_fraction_abs_max'].item() == 0.
            detail_valid &= detail['large_bbox_exact_baseline']
            detail_valid &= detail['small_medium_fraction_exact_e4']
            for group in ('small', 'medium', 'large'):
                rate = detail[f'{group}_sber_enabled_ratio']
                if rate is not None:
                    detail_valid &= float(rate) == (0. if group == 'large' else 1.)
        reports.append({key: value.detach().tolist() if isinstance(value, torch.Tensor) else value
                        for key, value in detail.items()})
    all_ok = torch.tensor(int(boxes_legal and gradients_finite and values_finite and detail_valid),
                          device=boxes.device)
    if dist_utils.is_dist_available_and_initialized():
        torch.distributed.all_reduce(all_ok, op=torch.distributed.ReduceOp.MIN)
    if dist_utils.is_main_process():
        print('SBER_FIRST_BATCH', {
            'global_step': global_step, 'layers': reports,
            'parameters_with_gradient': f'{sum(p.grad is not None for _, p in parameters)}/{len(parameters)}',
            'gradient_abs_max': {name: float(parameter.grad.detach().abs().max())
                                 if parameter.grad is not None else None for name, parameter in parameters},
            'refined_boxes_legal': boxes_legal,
            'box_range': [float(boxes.min()), float(boxes.max())],
            'corner_range': [float(corners.min()), float(corners.max())],
            'gradients_finite': gradients_finite, 'values_finite': values_finite,
            'all_ranks_passed': bool(all_ok.item()),
        }, flush=True)
    if not all_ok.item():
        raise FloatingPointError('SBER first-batch sampling/box/gradient/finite check failed on a rank')
    head._debug_done = True


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
            check_umqr_first_batch(model, criterion, outputs, targets, loss_dict, global_step)
            check_sber_first_batch(model, outputs, loss_dict, global_step)
            
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
            check_umqr_first_batch(model, criterion, outputs, targets, loss_dict, global_step)
            check_sber_first_batch(model, outputs, loss_dict, global_step)
            
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
def evaluate(model: torch.nn.Module, criterion: torch.nn.Module, postprocessor,
             data_loader, coco_evaluator: CocoEvaluator, device,
             query_diagnosis_output_dir=None, query_selection_metrics=False):
    model.eval()
    criterion.eval()
    coco_evaluator.cleanup()
    iou_types = coco_evaluator.iou_types

    metric_logger = MetricLogger(delimiter="  ")
    header = 'Test:'
    query_stats = None
    query_stats_skipped = False
    selection_stats = QuerySelectionMetrics() if query_selection_metrics else None
    probe_stats = None
    quality_diagnostic_mode = getattr(
        postprocessor, 'decoder_quality_iou_mode', 'class_agnostic')
    if quality_diagnostic_mode == 'predicted_class':
        probe_stats = FinalQualityProbeStats(query_diagnosis_output_dir)
    elif quality_diagnostic_mode == 'pairwise_class':
        probe_stats = ClassConditionedQualityProbeStats(
            query_diagnosis_output_dir)
    elif quality_diagnostic_mode == 'pairwise_class_ranking':
        probe_stats = ClassConditionedQualityProbeStats(
            query_diagnosis_output_dir, ranking_enhanced=True)
    elif quality_diagnostic_mode == 'multi_threshold_pairwise':
        probe_stats = MultiThresholdClassConditionedQualityProbeStats(
            query_diagnosis_output_dir)
    
    for samples, targets in metric_logger.log_every(data_loader, 10, header):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        outputs = model(samples)

        if selection_stats is not None:
            selection_stats.update(outputs, targets, samples.shape[-2:])

        if probe_stats is not None:
            probe_stats.update(outputs, targets, samples.shape[-2:])

        if 'enc_topk_boxes' in outputs and probe_stats is None:
            if dist_utils.get_world_size() == 1:
                if query_stats is None:
                    query_stats = QueryStats(
                        query_diagnosis_output_dir,
                        final_quality_gamma=getattr(
                            postprocessor, 'final_quality_gamma', 0.0))
                diagnostic_images = (
                    samples if query_diagnosis_output_dir is not None else None)
                query_stats.update(
                    outputs, targets, samples.shape[-2:], diagnostic_images)
            elif not query_stats_skipped:
                print('Encoder Top-K Query Diagnosis skipped: '
                      'the first implementation supports single-GPU evaluation only.')
                query_stats_skipped = True

        # TODO (lyuwenyu), fix dataset converted using `convert_to_coco_api`?
        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        
        oracle_enabled = (
            getattr(postprocessor, 'oracle_final_iou_gamma', 0.0) != 0.0
            or getattr(
                postprocessor,
                'class_aware_oracle_final_iou_gamma', 0.0) != 0.0
            or getattr(
                postprocessor,
                'pairwise_class_aware_oracle_beta', 0.0) != 0.0)
        if oracle_enabled:
            results = postprocessor(
                outputs, orig_target_sizes, targets=targets,
                image_hw=samples.shape[-2:])
        else:
            results = postprocessor(outputs, orig_target_sizes)
        if query_stats is not None:
            query_stats.update_final_predictions(
                results, targets, samples.shape[-2:])

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

    probe_summary = None
    if probe_stats is not None:
        probe_stats.synchronize_between_processes()
        main_process = dist_utils.is_main_process()
        probe_summary = probe_stats.summarize(
            write_output=main_process,
            print_output=main_process)

    # accumulate predictions from all images
    if coco_evaluator is not None:
        coco_evaluator.accumulate()
        coco_evaluator.summarize()

    if query_stats is not None:
        query_stats.summarize()

    stats = {}
    if selection_stats is not None:
        stats['query_selection_metrics'] = selection_stats.summarize()
    # stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    if coco_evaluator is not None:
        if 'bbox' in iou_types:
            stats['coco_eval_bbox'] = coco_evaluator.coco_eval['bbox'].stats.tolist()
        if 'segm' in iou_types:
            stats['coco_eval_masks'] = coco_evaluator.coco_eval['segm'].stats.tolist()
    if probe_summary is not None:
        stats[probe_stats.metric_name] = probe_summary.get(
            'checkpoint_metric_values',
            [probe_summary['pearson'], probe_summary['spearman']])
            
    return stats, coco_evaluator

