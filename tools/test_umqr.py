"""CPU synthetic correctness checks for Baseline + UMQR, not SAR results."""

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

from src.core import YAMLConfig
from src.core._config import BaseConfig
from src.core.yaml_utils import load_config
from src.optim import ModelEMA
from src.solver.det_engine import train_one_epoch, check_umqr_first_batch
from src.solver.det_solver import DetSolver
from src.zoo.rtdetr.umqr import UMQRHead, box_keypoints, uncertainty_keypoint_loss


ROOT = Path(__file__).resolve().parents[1]
E0 = ROOT / 'configs/rtdetrv2/rtdetrv2_r18vd_80e_ogsod_baseline_gbs64_gpu2_zxy.yml'
E3 = ROOT / 'configs/rtdetrv2/rtdetrv2_r18vd_80e_ogsod_umqr_gbs64_gpu2_zxy.yml'


def config(path=E3):
    # Like sar_audit.fresh_config: repeated configs in this test must not
    # share the existing YAML loader's mutable default (normal CLI loads once).
    cfg = YAMLConfig.__new__(YAMLConfig)
    BaseConfig.__init__(cfg)
    clean = load_config(str(path), cfg={})
    for key in list(cfg.__dict__):
        if not key.startswith('_') and key in clean:
            cfg.__dict__[key] = clean[key]
    cfg.yaml_cfg = copy.deepcopy(clean)
    cfg.yaml_cfg['PResNet']['pretrained'] = False
    # Synthetic image size only; keep 300 queries, three layers, baseline DN.
    cfg.yaml_cfg['eval_spatial_size'] = [160, 160]
    return cfg


def training_targets():
    return [
        {'labels': torch.tensor([0, 2]), 'boxes': torch.tensor([[.3, .4, .06, .08], [.6, .5, .3, .2]])},
        {'labels': torch.empty(0, dtype=torch.long), 'boxes': torch.empty(0, 4)},
    ]


def distributed_worker(rank, port):
    torch.set_num_threads(1)
    torch.distributed.init_process_group(
        'gloo', init_method=f'tcp://127.0.0.1:{port}?use_libuv=0', rank=rank, world_size=2)
    try:
        cfg = config()
        model = nn.parallel.DistributedDataParallel(cfg.model)
        optimizer = cfg.optimizer
        images = torch.rand(2, 3, 160, 160)
        targets = training_targets()
        # Unequal foreground counts across ranks, retaining active baseline DN
        # on each rank (baseline DN embedding is unused for entirely empty ranks).
        if rank == 1:
            targets[1] = {'labels': torch.tensor([1]), 'boxes': torch.tensor([[.5, .6, .1, .1]])}
        for step in range(2):
            optimizer.zero_grad()
            outputs = model(images, targets)
            losses = cfg.criterion(outputs, targets)
            sum(losses.values()).backward()
            check_umqr_first_batch(model, cfg.criterion, outputs, targets, losses, step)
            assert all(p.grad is not None and torch.isfinite(p.grad).all()
                       for p in model.module.decoder.umqr_head.parameters())
            optimizer.step()
    finally:
        torch.distributed.destroy_process_group()


class UMQRChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_five_points_and_uncertainty_numerics(self):
        boxes = torch.tensor([[.5, .5, .4, .2]])
        expected = torch.tensor([[[.5, .5], [.3, .5], [.7, .5], [.5, .4], [.5, .6]]])
        torch.testing.assert_close(box_keypoints(boxes), expected)
        self.assertEqual(box_keypoints(torch.empty(0, 4)).shape, (0, 5, 2))
        for value in (0., torch.log(torch.tensor(2.)).item(), -1e4, 1e4):
            points = (expected + .1).requires_grad_()
            scales = torch.full_like(points, value, requires_grad=True)
            loss = uncertainty_keypoint_loss(points, scales, expected, 1.)
            clamp = torch.tensor(value).clamp(-5., 3.)
            torch.testing.assert_close(loss, .1 * (-clamp).exp() + clamp)
            loss.backward()
            self.assertTrue(torch.isfinite(loss))
            self.assertTrue(torch.isfinite(points.grad).all() and torch.isfinite(scales.grad).all())
        loss = uncertainty_keypoint_loss(expected, torch.full_like(expected, -5.), expected, 1.)
        self.assertEqual(float(loss), -5.)  # Valid log-scale NLL; do not relu the loss.

        head = UMQRHead(16, 8)
        with torch.autocast('cpu', dtype=torch.bfloat16):
            refined, points, scales = head(torch.randn(2, 3, 16), boxes.expand(2, 3, 4))
            loss = uncertainty_keypoint_loss(points, scales, points.detach(), 6.) + refined.square().mean()
        self.assertEqual(points.dtype, torch.float32)
        self.assertEqual(scales.dtype, torch.float32)
        self.assertEqual(loss.dtype, torch.float32)
        loss.backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in head.parameters()))

    def test_foreground_only_and_empty(self):
        criterion = config().criterion
        points = torch.rand(1, 3, 5, 2, requires_grad=True)
        scales = torch.zeros_like(points, requires_grad=True)
        outputs = {'pred_boxes': torch.rand(1, 3, 4), 'pred_keypoints': points,
                   'pred_keypoint_log_scales': scales}
        targets = [{'boxes': torch.tensor([[.5, .5, .2, .2]])}]
        indices = [(torch.tensor([1]), torch.tensor([0]))]
        loss = criterion.loss_umqr(outputs, targets, indices, 1.)['loss_ukp']
        loss.backward()
        self.assertGreater(float(points.grad[:, 1].abs().sum()), 0.)
        self.assertEqual(float(points.grad[:, [0, 2]].abs().sum()), 0.)
        self.assertEqual(float(scales.grad[:, [0, 2]].abs().sum()), 0.)
        points.grad = scales.grad = None
        targets = [{'boxes': torch.empty(0, 4)}]
        indices = [(torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long))]
        empty = criterion.loss_umqr(outputs, targets, indices, 1.)['loss_ukp']
        self.assertEqual(float(empty), 0.)
        empty.backward()
        self.assertIsNotNone(points.grad)
        self.assertIsNotNone(scales.grad)
        self.assertEqual(float(points.grad.abs().sum() + scales.grad.abs().sum()), 0.)

    def test_baseline_isolation_training_geometry_checkpoint(self):
        baseline_cfg, umqr_cfg = load_config(str(E0), cfg={}), load_config(str(E3), cfg={})
        expected = copy.deepcopy(baseline_cfg)
        expected.update(output_dir=umqr_cfg['output_dir'], umqr_evaluation_metrics=True,
                        umqr_log_scale_min=-5.0, umqr_log_scale_max=3.0)
        expected['__include__'] = umqr_cfg['__include__']
        self.assertEqual(umqr_cfg['__include__'], ['./rtdetrv2_r18vd_80e_ogsod_baseline_gbs64_gpu2_zxy.yml'])
        expected['RTDETRTransformerv2'].update(umqr=True, umqr_hidden_dim=64)
        expected['RTDETRCriterionv2'].update(umqr=True)
        expected['RTDETRCriterionv2']['weight_dict']['loss_ukp'] = 1.0
        self.assertEqual(expected, umqr_cfg)
        torch.manual_seed(0)
        e0 = config(E0)
        baseline = e0.model
        rng = torch.get_rng_state()
        torch.manual_seed(0)
        cfg = config()
        model, criterion, optimizer = cfg.model, cfg.criterion, cfg.optimizer
        torch.testing.assert_close(rng, torch.get_rng_state())
        for name, value in baseline.state_dict().items():
            torch.testing.assert_close(value, model.state_dict()[name], rtol=0, atol=0)
        head = model.decoder.umqr_head
        self.assertEqual(sum(p.numel() for p in head.parameters()), 35732)
        self.assertEqual(model.decoder.query_select_method, 'default')
        self.assertFalse(model.decoder.sar_quality)
        self.assertFalse(any('quality_head' in name for name, _ in model.named_parameters()))
        self.assertEqual(model.decoder.num_queries, 300)
        self.assertEqual(model.decoder.num_layers, 3)
        self.assertEqual(cfg.yaml_cfg['train_dataloader']['total_batch_size'], 64)
        optimizer_ids = {id(p) for group in optimizer.param_groups for p in group['params']}
        self.assertTrue(all(id(p) in optimizer_ids for p in head.parameters()))

        images, targets = torch.rand(2, 3, 160, 160), training_targets()
        baseline.eval()
        model.eval()
        with torch.no_grad():
            before, baseline_output = model(images), baseline(images)
        for key in ('pred_boxes', 'pred_logits', 'enc_topk_boxes', 'enc_topk_indices'):
            torch.testing.assert_close(before[key], baseline_output[key], rtol=0, atol=0)

        baseline.train()
        model.train()
        torch.manual_seed(12)
        baseline_output = baseline(images, targets)
        torch.manual_seed(12)
        output = model(images, targets)
        torch.testing.assert_close(output['pred_boxes'], baseline_output['pred_boxes'], rtol=0, atol=0)
        torch.testing.assert_close(output['pred_logits'], baseline_output['pred_logits'], rtol=0, atol=0)
        self.assertEqual(output['pred_keypoints'].shape, (2, 300, 5, 2))
        self.assertEqual(output['pred_keypoint_log_scales'].shape, (2, 300, 5, 2))
        self.assertEqual(len(output['aux_outputs']), 2)
        self.assertTrue(all(o['pred_keypoints'].shape == (2, 300, 5, 2) for o in output['aux_outputs']))
        self.assertTrue(all('pred_keypoints' not in o for o in output['dn_aux_outputs']))
        self.assertTrue(all('pred_keypoints' not in o for o in output['enc_aux_outputs']))
        counts = [0, 0]
        def count_matcher(index):
            def hook(*_):
                counts[index] += 1
            return hook
        hooks = [e0.criterion.matcher.register_forward_hook(count_matcher(0)),
                 criterion.matcher.register_forward_hook(count_matcher(1))]
        self.assertFalse(e0.criterion.umqr)
        self.assertTrue(criterion.umqr)
        base_loss, augmented_loss = e0.criterion(baseline_output, targets), criterion(output, targets)
        for hook in hooks:
            hook.remove()
        self.assertEqual(counts, [4, 4])
        for key, value in base_loss.items():
            torch.testing.assert_close(value, augmented_loss[key], rtol=0, atol=0)
        self.assertEqual(set(augmented_loss) - set(base_loss),
                         {'loss_ukp', 'loss_ukp_aux_0', 'loss_ukp_aux_1'})
        del output, baseline_output, augmented_loss, base_loss, baseline

        ema = ModelEMA(model)
        log = io.StringIO()
        with contextlib.redirect_stdout(log):
            stats = train_one_epoch(model, criterion, [(images, targets)] * 3,
                                    optimizer, torch.device('cpu'), 0, ema=ema, max_norm=.1)
        self.assertEqual(log.getvalue().count('UMQR_FIRST_BATCH'), 1)
        self.assertIn("'all_ranks_passed': True", log.getvalue())
        self.assertTrue(all(torch.isfinite(torch.tensor(value)) for value in stats.values()))

        # Detection-only gradients establish an active (not auxiliary-only) path.
        optimizer.zero_grad()
        outputs = model(images, targets)
        outputs['pred_boxes'].sum().backward()
        gradient = head.prediction[-1].weight.grad
        self.assertGreater(float(gradient[:10].abs().sum()), 0.)
        self.assertGreater(float(gradient[10:].abs().sum()), 0.)
        self.assertTrue(all(p.grad is not None for p in head.geometry.parameters()))
        optimizer.zero_grad()
        empty_targets = [{'labels': torch.empty(0, dtype=torch.long), 'boxes': torch.empty(0, 4)}] * 2
        empty_losses = criterion(model(images, empty_targets), empty_targets)
        sum(empty_losses.values()).backward()
        self.assertEqual(float(empty_losses['loss_ukp']), 0.)
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in head.parameters()))

        model.eval()
        with torch.no_grad():
            original = model(images)
            bias = head.prediction[-1].bias.clone()
            head.prediction[-1].bias[:10].add_(.6)
            changed_points = model(images)
            head.prediction[-1].bias.copy_(bias)
            head.prediction[-1].bias[10:].add_(.5)
            changed_scales = model(images)
            head.prediction[-1].bias.copy_(bias)
        for intervention in (changed_points, changed_scales):
            self.assertGreater(float((original['pred_boxes'] - intervention['pred_boxes']).abs().max()), 0.)
            torch.testing.assert_close(original['enc_topk_indices'], intervention['enc_topk_indices'])

        stream = io.BytesIO()
        torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                    'ema': ema.state_dict(), 'criterion': criterion.state_dict()}, stream)
        stream.seek(0)
        state = torch.load(stream, weights_only=True)
        restored = config()
        restored.model.load_state_dict(state['model'], strict=True)
        restored.optimizer.load_state_dict(state['optimizer'])
        restored.criterion.load_state_dict(state['criterion'], strict=True)
        restored_ema = ModelEMA(restored.model)
        restored_ema.load_state_dict(state['ema'], strict=True)
        for parameter in restored.model.decoder.umqr_head.parameters():
            self.assertIn('exp_avg', restored.optimizer.state[parameter])
        restored.model.eval()
        with torch.no_grad():
            torch.testing.assert_close(original['pred_boxes'], restored.model(images)['pred_boxes'])
        torch.testing.assert_close(ema.module.decoder.umqr_head.geometry[-1].weight,
                                   restored_ema.module.decoder.umqr_head.geometry[-1].weight)

    def test_full_coco_eval_ema_metrics_and_query_diagnosis(self):
        from PIL import Image
        with tempfile.TemporaryDirectory(prefix='umqr_eval_') as directory:
            root = Path(directory)
            images, annotations = [], []
            for i in range(3):
                Image.new('RGB', (160, 160), (80, 100, 120)).save(root / f'{i}.png')
                images.append({'id': i, 'file_name': f'{i}.png', 'width': 160, 'height': 160})
                annotations.append({'id': i, 'image_id': i, 'category_id': i,
                                    'bbox': [30, 40, 8, 10], 'area': 80, 'iscrowd': 0})
            path = root / 'val.json'
            path.write_text(json.dumps({'images': images, 'annotations': annotations,
                                       'categories': [{'id': i, 'name': str(i)} for i in range(3)]}))
            cfg = config()
            cfg.device, cfg.output_dir = 'cpu', str(root / 'evaluation')
            cfg.yaml_cfg['seed'] = 0
            loader = cfg.yaml_cfg['val_dataloader']
            loader.update(num_workers=0, total_batch_size=2)
            loader['dataset'].update(img_folder=str(root), ann_file=str(path))
            for op in loader['dataset']['transforms']['ops']:
                if op['type'] == 'Resize':
                    op['size'] = [160, 160]
            checkpoint = root / 'synthetic.pth'
            torch.save({'model': cfg.model.state_dict(), 'ema': cfg.ema.state_dict(),
                        'criterion': cfg.criterion.state_dict(), 'last_epoch': 0}, checkpoint)
            cfg.resume = str(checkpoint)
            log = io.StringIO()
            with contextlib.redirect_stdout(log):
                solver = DetSolver(cfg)
                solver.val()
            output = Path(cfg.output_dir)
            focused = json.loads((output / 'umqr_metrics.json').read_text())
            coco = json.loads((output / 'evaluation_metrics.json').read_text())
            for key in ('AP', 'AP50', 'AP75', 'APs', 'AR100', 'AR75'):
                self.assertEqual(focused[key], coco[key])
            self.assertEqual(focused['weights'], 'ema')
            self.assertEqual(focused['checkpoint'], str(checkpoint))
            self.assertEqual(focused['config']['seed'], 0)
            # Independently index the COCO recall tensor for AR@0.75.
            import numpy as np
            evaluation = solver.evaluator.coco_eval['bbox']
            p = evaluation.params
            recall = evaluation.eval['recall'][np.flatnonzero(np.isclose(p.iouThrs, .75))[0],
                                               :, list(p.areaRngLbl).index('all'), list(p.maxDets).index(100)]
            self.assertAlmostEqual(focused['AR75'], float(recall[recall > -1].mean()))
            self.assertTrue((output / 'eval.pth').exists())
            self.assertIn('Encoder Top-K Query Diagnosis', log.getvalue())
            self.assertTrue((output / 'query_diagnosis').is_dir())
            if solver.writer:
                solver.writer.close()

    def test_two_process_full_model_ddp(self):
        if not torch.distributed.is_gloo_available():
            self.skipTest('Gloo unavailable')
        with socket.socket() as socket_:
            socket_.bind(('127.0.0.1', 0))
            port = socket_.getsockname()[1]
        torch.multiprocessing.spawn(distributed_worker, args=(port,), nprocs=2, join=True)


if __name__ == '__main__':
    unittest.main(verbosity=2)
