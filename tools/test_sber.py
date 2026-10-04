"""Focused E4 synthetic verification; does not establish real SAR AP."""

import contextlib
import copy
import io
import json
from pathlib import Path
import socket
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from torch import nn

from tools.test_umqr import ROOT, E0, config, training_targets
from src.core.yaml_utils import load_config
from src.optim import ModelEMA
from src.solver.det_engine import train_one_epoch, check_sber_first_batch
from src.solver.det_solver import DetSolver
from src.zoo.rtdetr.box_ops import box_cxcywh_to_xyxy
from src.zoo.rtdetr.sber import (
    SBERHead, encoder_feature_maps, boundary_sampling_coordinates,
    box_scale_weights, sample_boundary_features, apply_boundary_residual)

E4 = ROOT / 'configs/rtdetrv2/rtdetrv2_r18vd_80e_ogsod_sber_gbs64_gpu2_zxy.yml'


def analytic_maps(channels=2):
    maps = []
    for level, side in enumerate((80, 40, 20)):
        y, x = torch.meshgrid((torch.arange(side) + .5) / side,
                              (torch.arange(side) + .5) / side, indexing='ij')
        maps.append((2 * x + 3 * y + 10 * level)[None, None].repeat(1, channels, 1, 1))
    return maps


def assert_legal(boxes):
    assert torch.isfinite(boxes).all()
    assert ((boxes >= 0) & (boxes <= 1)).all()
    assert (boxes[..., 2:] > 0).all()
    corners = box_cxcywh_to_xyxy(boxes)
    assert ((corners >= -1e-6) & (corners <= 1 + 1e-6)).all()


def distributed_worker(rank, port):
    torch.set_num_threads(1)
    torch.distributed.init_process_group(
        'gloo', init_method=f'tcp://127.0.0.1:{port}?use_libuv=0', rank=rank, world_size=2)
    try:
        cfg = config(E4)
        model = nn.parallel.DistributedDataParallel(cfg.model)
        images, targets = torch.rand(2, 3, 160, 160), training_targets()
        if rank == 1:
            targets[1] = {'labels': torch.tensor([1]), 'boxes': torch.tensor([[.6, .5, .1, .1]])}
        for step in range(2):
            cfg.optimizer.zero_grad()
            output = model(images, targets)
            losses = cfg.criterion(output, targets)
            sum(losses.values()).backward()
            check_sber_first_batch(model, output, losses, step)
            assert all(p.grad is not None and torch.isfinite(p.grad).all()
                       for p in model.module.decoder.sber_head.parameters())
            assert_legal(output['pred_boxes'])
            cfg.optimizer.step()
    finally:
        torch.distributed.destroy_process_group()


class SBERChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_real_sampling_coordinates_scale_fusion_and_feature_gradient(self):
        refs = torch.tensor([[[.5, .5, .05, .05], [.5, .5, .1, .1], [.5, .5, .2, .2]]])
        maps = analytic_maps()
        coordinates = boundary_sampling_coordinates(refs)
        self.assertEqual(coordinates.shape, (1, 3, 4, 3, 2))
        torch.testing.assert_close(coordinates[0, 1, 0],
                                   torch.tensor([[.46, .5], [.45, .5], [.44, .5]]))
        torch.testing.assert_close(coordinates[0, 1, 1],
                                   torch.tensor([[.54, .5], [.55, .5], [.56, .5]]))
        torch.testing.assert_close(coordinates[0, 1, 2],
                                   torch.tensor([[.5, .46], [.5, .45], [.5, .44]]))
        weights = box_scale_weights(refs, [(80, 80), (40, 40), (20, 20)])
        torch.testing.assert_close(weights[0], torch.eye(3))
        middle = refs[:, :1].clone()
        middle[..., 2:] = .05 * 2 ** .5
        torch.testing.assert_close(box_scale_weights(middle, [(80, 80), (40, 40), (20, 20)]),
                                   torch.tensor([[[.5, .5, 0.]]]))
        sampled = sample_boundary_features(maps, coordinates, weights)
        expected = 2 * coordinates[..., 0] + 3 * coordinates[..., 1]
        expected += torch.tensor([0., 10., 20.])[None, :, None, None]
        torch.testing.assert_close(sampled[..., 0], expected, atol=5e-6, rtol=1e-6)
        contrast = sampled[..., 0, 0] - sampled[..., 2, 0]
        expected_contrast = refs[..., 2:3] * torch.tensor([.4, -.4, .6, -.6])
        torch.testing.assert_close(contrast, expected_contrast, atol=5e-6, rtol=1e-5)
        # Real grid_sample has a live path into every selected encoder map.
        maps = [feature.clone().requires_grad_() for feature in maps]
        head = SBERHead(2, 8)
        nn.init.normal_(head.offset_head.weight, std=.03)
        nn.init.constant_(head.gate_head.weight, .02)
        fractions, details = head(maps, refs, collect_debug=True)
        apply_boundary_residual(refs, fractions).sum().backward()
        self.assertTrue(all(feature.grad is not None and feature.grad.abs().sum() > 0 for feature in maps))
        self.assertGreater(float(details['contrast_abs_mean']), 0.)
        altered, _ = head([torch.zeros_like(feature) for feature in maps], refs)
        self.assertGreater(float((fractions - altered).abs().max()), 0.)
        memory = torch.cat([feature.detach().flatten(2).transpose(1, 2) for feature in maps], dim=1)
        restored = encoder_feature_maps(memory, [(80, 80), (40, 40), (20, 20)])
        for original, recovered in zip(maps, restored):
            torch.testing.assert_close(original, recovered, rtol=0, atol=0)

    def test_bounded_edges_legality_extremes_and_autocast(self):
        boxes = torch.tensor([[[.5, .5, .4, .2], [.001, .999, 1., 1.], [.5, .5, 1e-8, 1e-8]]])
        fractions = torch.tensor([[[.1, -.1, -.1, .1]]]).expand_as(boxes)
        refined = apply_boundary_residual(boxes, fractions)
        torch.testing.assert_close(refined[:, :1], torch.tensor([[[.5, .5, .32, .24]]]))
        assert_legal(refined)
        coordinates = boundary_sampling_coordinates(boxes)
        self.assertTrue(((coordinates >= 0) & (coordinates <= 1)).all())
        head = SBERHead(2, 8)
        self.assertTrue((head.offset_head.weight == 0).all())
        for value in (-1e4, 0., 1e4):
            with torch.no_grad():
                head.gate_head.bias.fill_(value)
                head.offset_head.bias.fill_(value)
            with torch.autocast('cpu', dtype=torch.bfloat16):
                offsets, details = head(analytic_maps(), boxes, collect_debug=True)
                prediction = apply_boundary_residual(boxes, offsets)
            self.assertEqual(offsets.dtype, torch.float32)
            self.assertLessEqual(float(offsets.abs().max()), .1 + 1e-6)
            self.assertTrue(0 < float(details['gate_min']) <= float(details['gate_max']) < 1)
            assert_legal(prediction)
            prediction.sum().backward()
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in head.parameters()))
            head.zero_grad()

    def test_config_disabled_baseline_training_and_checkpoint(self):
        baseline_cfg, sber_cfg = load_config(str(E0), cfg={}), load_config(str(E4), cfg={})
        expected = copy.deepcopy(baseline_cfg)
        expected['__include__'] = ['./rtdetrv2_r18vd_80e_ogsod_baseline_gbs64_gpu2_zxy.yml']
        expected['output_dir'] = sber_cfg['output_dir']
        expected['sber_evaluation_metrics'] = True
        expected['RTDETRTransformerv2'].update(sber=True, sber_hidden_dim=64, sber_rho=.1)
        self.assertEqual(expected, sber_cfg)
        torch.manual_seed(0)
        baseline = config(E0).model
        rng = torch.get_rng_state()
        torch.manual_seed(0)
        cfg = config(E4)
        model, criterion, optimizer = cfg.model, cfg.criterion, cfg.optimizer
        torch.testing.assert_close(rng, torch.get_rng_state())
        head = model.decoder.sber_head
        self.assertEqual(sum(p.numel() for p in head.parameters()), 131461)
        self.assertFalse(hasattr(model.decoder, 'umqr_head'))
        self.assertFalse(hasattr(model.decoder, 'enc_quality_head'))
        self.assertFalse(criterion.umqr)
        self.assertEqual(criterion.weight_dict, {'loss_vfl': 1, 'loss_bbox': 5, 'loss_giou': 2})
        for name, value in baseline.state_dict().items():
            torch.testing.assert_close(value, model.state_dict()[name], rtol=0, atol=0)
        self.assertTrue(all(any(p is q for group in optimizer.param_groups for q in group['params'])
                            for p in head.parameters()))

        disabled_cfg = config(E4)
        disabled_cfg.yaml_cfg['RTDETRTransformerv2']['sber'] = False
        disabled = disabled_cfg.model
        self.assertEqual(set(disabled.state_dict()), set(baseline.state_dict()))
        with torch.no_grad():
            for bbox in baseline.decoder.dec_bbox_head:
                nn.init.normal_(bbox.layers[-1].weight, std=.02)
        disabled.load_state_dict(baseline.state_dict(), strict=True)
        images, targets = torch.rand(2, 3, 160, 160), training_targets()
        baseline.eval()
        disabled.eval()
        with torch.no_grad():
            original, predicted = baseline(images), disabled(images)
        self.assertEqual(set(original), set(predicted))
        for key in original:
            torch.testing.assert_close(original[key], predicted[key], rtol=0, atol=0)
        baseline.train()
        disabled.train()
        state = torch.get_rng_state()
        original = baseline(images, targets)
        torch.set_rng_state(state)
        predicted = disabled(images, targets)
        for key in ('pred_boxes', 'pred_logits'):
            torch.testing.assert_close(original[key], predicted[key], rtol=0, atol=0)
        first_losses, second_losses = config(E0).criterion(original, targets), disabled_cfg.criterion(predicted, targets)
        self.assertEqual(set(first_losses), set(second_losses))
        for key in first_losses:
            torch.testing.assert_close(first_losses[key], second_losses[key], rtol=0, atol=0)

        calls = []
        hook = criterion.matcher.register_forward_hook(lambda *args: calls.append(1))
        log = io.StringIO()
        with contextlib.redirect_stdout(log):
            stats = train_one_epoch(model, criterion, [(images, targets)] * 3,
                                    optimizer, torch.device('cpu'), 0, max_norm=.1)
        hook.remove()
        self.assertEqual(len(calls), 3 * 4)  # final + two decoder aux + encoder; no new matches.
        self.assertEqual(set(stats) - {'lr', 'loss'}, set(first_losses))
        self.assertEqual(log.getvalue().count('SBER_FIRST_BATCH'), 1)
        self.assertIn("'parameters_with_gradient': '6/6'", log.getvalue())
        self.assertIn("'all_ranks_passed': True", log.getvalue())
        self.assertTrue(all(torch.isfinite(torch.tensor(value)) for value in stats.values()))
        optimizer.zero_grad()
        losses = criterion(model(images, targets), targets)
        sum(losses.values()).backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0
                            for p in head.parameters()))
        model.eval()
        with torch.no_grad():
            original = model(images)
            hook = head.register_forward_pre_hook(
                lambda module, args: ([torch.zeros_like(feature) for feature in args[0]], args[1]))
            altered = model(images)
            hook.remove()
        self.assertGreater(float((original['pred_boxes'] - altered['pred_boxes']).abs().max()), 0.)
        torch.testing.assert_close(original['enc_topk_indices'], altered['enc_topk_indices'])
        self.assertNotIn('pred_keypoints', original)
        self.assertNotIn('sber_debug', original)
        ema = ModelEMA(model, decay=.9999, warmups=2000)
        ema.update(model)
        stream = io.BytesIO()
        torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'ema': ema.state_dict()}, stream)
        stream.seek(0)
        state = torch.load(stream, weights_only=True)
        recovered = config(E4)
        recovered.model.load_state_dict(state['model'], strict=True)
        recovered.optimizer.load_state_dict(state['optimizer'])
        recovered.ema.load_state_dict(state['ema'])
        self.assertIn('exp_avg', recovered.optimizer.state[recovered.model.decoder.sber_head.offset_head.weight])
        recovered.model.eval()
        with torch.no_grad():
            torch.testing.assert_close(original['pred_boxes'], recovered.model(images)['pred_boxes'])
        for name, value in ema.module.decoder.sber_head.state_dict().items():
            torch.testing.assert_close(value, recovered.ema.module.decoder.sber_head.state_dict()[name])

    def test_actual_solver_fit_best_eval_and_original_diagnosis(self):
        from PIL import Image
        with tempfile.TemporaryDirectory(prefix='sber_') as directory:
            root = Path(directory)
            images, annotations = [], []
            for i in range(2):
                Image.new('RGB', (160, 160), (80, 100, 120)).save(root / f'{i}.png')
                images.append({'id': i, 'file_name': f'{i}.png', 'width': 160, 'height': 160})
                annotations.append({'id': i, 'image_id': i, 'category_id': i,
                                    'bbox': [60, 60, 8, 10], 'area': 80, 'iscrowd': 0})
            path = root / 'fixture.json'
            path.write_text(json.dumps({'images': images, 'annotations': annotations,
                                       'categories': [{'id': i, 'name': str(i)} for i in range(3)]}))
            def fixture_config(output):
                cfg = config(E4)
                cfg.device, cfg.output_dir = 'cpu', str(output)
                cfg.yaml_cfg['seed'] = 0
                for name in ('train_dataloader', 'val_dataloader'):
                    loader = cfg.yaml_cfg[name]
                    loader.update(num_workers=0, total_batch_size=2)
                    loader['dataset'].update(img_folder=str(root), ann_file=str(path))
                    for op in loader['dataset']['transforms']['ops']:
                        if op['type'] == 'Resize':
                            op['size'] = [160, 160]
                return cfg
            cfg = fixture_config(root / 'train')
            cfg.epoches = 1  # synthetic smoke only, not an experiment change.
            log = io.StringIO()
            with contextlib.redirect_stdout(log):
                training = DetSolver(cfg)
                training.fit()
            self.assertIn('SBER_FIRST_BATCH', log.getvalue())
            self.assertTrue((Path(cfg.output_dir) / 'last.pth').exists())
            self.assertTrue((Path(cfg.output_dir) / 'best.pth').exists())
            if training.writer:
                training.writer.close()
            evaluation_cfg = fixture_config(root / 'evaluation')
            evaluation_cfg.resume = str(Path(cfg.output_dir) / 'best.pth')
            log = io.StringIO()
            with contextlib.redirect_stdout(log):
                evaluation = DetSolver(evaluation_cfg)
                evaluation.val()
            if evaluation.writer:
                evaluation.writer.close()
            output = Path(evaluation_cfg.output_dir)
            focused = json.loads((output / 'sber_metrics.json').read_text())
            full = json.loads((output / 'evaluation_metrics.json').read_text())
            for key in ('AP', 'AP50', 'AP75', 'APs', 'APm', 'APl', 'AR100', 'AR75'):
                self.assertEqual(focused[key], full[key])
            import numpy as np
            coco = evaluation.evaluator.coco_eval['bbox']
            params = coco.params
            recall = coco.eval['recall'][np.flatnonzero(np.isclose(params.iouThrs, .75))[0],
                                         :, params.areaRngLbl.index('all'), params.maxDets.index(100)]
            self.assertAlmostEqual(focused['AR75'], float(recall[recall > -1].mean()))
            self.assertEqual(focused['weights'], 'ema')
            self.assertEqual(focused['checkpoint'], evaluation_cfg.resume)
            self.assertTrue((output / 'eval.pth').exists())
            self.assertIn('Encoder Top-K Query Diagnosis', log.getvalue())

    def test_two_process_full_model_ddp(self):
        if not torch.distributed.is_gloo_available():
            self.skipTest('Gloo unavailable')
        with socket.socket() as socket_:
            socket_.bind(('127.0.0.1', 0))
            port = socket_.getsockname()[1]
        torch.multiprocessing.spawn(distributed_worker, args=(port,), nprocs=2, join=True)


if __name__ == '__main__':
    unittest.main(verbosity=2)
