"""Synthetic E3' checks; no real SAR metrics or pretrained download."""

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

from tools.test_umqr import ROOT, E0, E3, config, training_targets
from src.core.yaml_utils import load_config
from src.solver.det_engine import train_one_epoch, check_umqr_first_batch
from src.solver.det_solver import DetSolver
from src.zoo.rtdetr.umqr import UMQRHead, box_keypoints, keypoints_to_box, uncertainty_keypoint_loss
from src.zoo.rtdetr.box_ops import box_cxcywh_to_xyxy

DIRECT = ROOT / 'configs/rtdetrv2/rtdetrv2_r18vd_80e_ogsod_umqr_direct_gbs64_gpu2_zxy.yml'


def distributed_worker(rank, port):
    torch.set_num_threads(1)
    torch.distributed.init_process_group(
        'gloo', init_method=f'tcp://127.0.0.1:{port}?use_libuv=0', rank=rank, world_size=2)
    try:
        cfg = config(DIRECT)
        model = nn.parallel.DistributedDataParallel(cfg.model)
        images, targets = torch.rand(2, 3, 160, 160), training_targets()
        if rank == 1:
            targets[1] = {'labels': torch.tensor([1]), 'boxes': torch.tensor([[.6, .5, .1, .1]])}
        for step in range(2):
            cfg.optimizer.zero_grad()
            outputs = model(images, targets)
            losses = cfg.criterion(outputs, targets)
            sum(losses.values()).backward()
            check_umqr_first_batch(model, cfg.criterion, outputs, targets, losses, step)
            head = model.module.decoder.umqr_head
            assert head.alpha.grad is not None and torch.isfinite(head.alpha.grad)
            assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in head.prediction.parameters())
            cfg.optimizer.step()
    finally:
        torch.distributed.destroy_process_group()


class UMQRDirectChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_geometry_legality_confidence_and_zero_alpha_gradients(self):
        reference = torch.tensor([[[.5, .5, .4, .2]]])
        points = box_keypoints(reference)
        torch.testing.assert_close(keypoints_to_box(points), reference)
        points[..., 0, 0] = .7
        torch.testing.assert_close(keypoints_to_box(points), torch.tensor([[[.6, .5, .4, .2]]]))
        extreme = torch.tensor([[[[2., -1.], [.9, .5], [.1, .5], [.5, .9], [.5, .1]]]])
        boxes = keypoints_to_box(extreme)
        self.assertTrue((boxes[..., 2:] > 0).all())
        corners = box_cxcywh_to_xyxy(boxes)
        self.assertTrue(((corners >= 0) & (corners <= 1)).all())
        for extreme in (torch.zeros(2, 3, 5, 2), torch.ones(2, 3, 5, 2)):
            boxes = keypoints_to_box(extreme)
            self.assertTrue(torch.isfinite(boxes).all())
            corners = box_cxcywh_to_xyxy(boxes)
            self.assertTrue(((corners >= 0) & (corners <= 1)).all())

        head = UMQRHead(16, 8, direct_box=True)
        self.assertFalse(hasattr(head, 'geometry'))
        self.assertEqual(float(head.alpha), 0.)
        for value in (-1e4, 0., 1e4):
            residual, _, confidence = head.box_residual(
                points, torch.full_like(points, value), reference)
            expected = torch.tensor(-value).clamp(-3., 5.).sigmoid()
            torch.testing.assert_close(confidence, expected.expand(1, 1, 1))
            torch.testing.assert_close(residual, torch.zeros_like(residual))
        # An offset makes alpha's detection gradient nonzero even at alpha=0.
        with torch.no_grad():
            head.prediction[-1].bias[0] = .4
        features = torch.randn(1, 1, 16)
        refined, p, s = head(features, reference)
        torch.testing.assert_close(refined, features, rtol=0, atol=0)
        residual, _, _ = head.box_residual(p, s, reference)
        detection = residual.sum()
        keypoint = uncertainty_keypoint_loss(p, s, box_keypoints(reference * .8), 1.)
        (detection + keypoint).backward()
        self.assertGreater(float(head.alpha.grad.abs()), 0.)
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in head.prediction.parameters()))
        self.assertGreater(float(head.prediction[-1].weight.grad[:10].abs().sum()), 0.)
        self.assertGreater(float(head.prediction[-1].weight.grad[10:].abs().sum()), 0.)
        head.zero_grad()
        with torch.autocast('cpu', dtype=torch.bfloat16):
            _, p, s = head(features, reference)
            residual, b, c = head.box_residual(p, s, reference)
            loss = residual.sum() + uncertainty_keypoint_loss(p, s, box_keypoints(reference), 1.)
        self.assertEqual(residual.dtype, torch.float32)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(all(torch.isfinite(tensor).all() for tensor in (b, c, head.alpha.grad)))

    def test_config_baseline_equivalence_and_training_gradient_path(self):
        e3 = load_config(str(E3), cfg={})
        direct = load_config(str(DIRECT), cfg={})
        expected = copy.deepcopy(e3)
        expected['output_dir'] = direct['output_dir']
        expected['__include__'] = ['./rtdetrv2_r18vd_80e_ogsod_umqr_gbs64_gpu2_zxy.yml']
        expected['RTDETRTransformerv2']['umqr_refinement'] = 'direct'
        self.assertEqual(expected, direct)
        torch.manual_seed(0)
        baseline = config(E0).model
        baseline_rng = torch.get_rng_state()
        torch.manual_seed(0)
        cfg = config(DIRECT)
        model, criterion, optimizer = cfg.model, cfg.criterion, cfg.optimizer
        torch.testing.assert_close(baseline_rng, torch.get_rng_state())
        head = model.decoder.umqr_head
        self.assertEqual(sum(p.numel() for p in head.parameters()), 17749)
        self.assertFalse(hasattr(head, 'geometry'))
        self.assertEqual(float(head.alpha), 0.)
        self.assertTrue(any(head.alpha is p for group in optimizer.param_groups for p in group['params']))
        for name, value in baseline.state_dict().items():
            torch.testing.assert_close(value, model.state_dict()[name], rtol=0, atol=0)
        images, targets = torch.rand(2, 3, 160, 160), training_targets()
        baseline.eval()
        model.eval()
        # Stronger than zero bbox initialization: alpha=0 stays equivalent
        # with arbitrary nonzero original bbox weights and offset keypoints.
        with torch.no_grad():
            for bbox in baseline.decoder.dec_bbox_head:
                nn.init.normal_(bbox.layers[-1].weight, std=.015)
            state = model.state_dict()
            state.update(baseline.state_dict())
            model.load_state_dict(state, strict=True)
            head.prediction[-1].bias[:10].add_(.3)
            original, predicted = baseline(images), model(images)
        for key in ('pred_boxes', 'pred_logits', 'enc_topk_indices'):
            torch.testing.assert_close(original[key], predicted[key], rtol=0, atol=0)

        log = io.StringIO()
        with contextlib.redirect_stdout(log):
            stats = train_one_epoch(model, criterion, [(images, targets)] * 3,
                                    optimizer, torch.device('cpu'), 0, max_norm=.1)
        self.assertEqual(log.getvalue().count('UMQR_FIRST_BATCH'), 1)
        self.assertIn("'B_kp_legal': True", log.getvalue())
        self.assertIn("'alpha_has_gradient': True", log.getvalue())
        self.assertIn("'all_ranks_passed': True", log.getvalue())
        self.assertTrue(all(torch.isfinite(torch.tensor(v)) for v in stats.values()))
        self.assertGreater(float(head.alpha.detach().abs()), 0.)
        optimizer.zero_grad()
        # At positive alpha, detection alone must train both p and s.
        with torch.no_grad():
            head.alpha.fill_(.25)
            head.prediction[-1].bias[0].add_(.3)
        model(images, targets)['pred_boxes'].sum().backward()
        self.assertGreater(float(head.prediction[-1].weight.grad[:10].abs().sum()), 0.)
        self.assertGreater(float(head.prediction[-1].weight.grad[10:].abs().sum()), 0.)
        self.assertIsNotNone(head.alpha.grad)
        model.eval()
        with torch.no_grad():
            original = model(images)
            bias = head.prediction[-1].bias.clone()
            head.prediction[-1].bias[:10].add_(.5)
            point_changed = model(images)
            head.prediction[-1].bias.copy_(bias)
            head.prediction[-1].bias[10:].add_(.5)
            scale_changed = model(images)
            head.prediction[-1].bias.copy_(bias)
        for changed in (point_changed, scale_changed):
            self.assertGreater(float((original['pred_boxes'] - changed['pred_boxes']).abs().max()), 0.)
            torch.testing.assert_close(original['enc_topk_indices'], changed['enc_topk_indices'])
        stream = io.BytesIO()
        torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict()}, stream)
        stream.seek(0)
        state = torch.load(stream, weights_only=True)
        restored = config(DIRECT)
        restored.model.load_state_dict(state['model'], strict=True)
        restored.optimizer.load_state_dict(state['optimizer'])
        self.assertIn('exp_avg', restored.optimizer.state[restored.model.decoder.umqr_head.alpha])
        restored.model.eval()
        with torch.no_grad():
            torch.testing.assert_close(original['pred_boxes'], restored.model(images)['pred_boxes'])

    def test_fit_final_alpha_and_best_eval_ema_metrics(self):
        from PIL import Image
        with tempfile.TemporaryDirectory(prefix='umqr_direct_') as directory:
            root = Path(directory)
            images, annotations = [], []
            for i in range(4):
                Image.new('RGB', (160, 160), (80, 100, 120)).save(root / f'{i}.png')
                images.append({'id': i, 'file_name': f'{i}.png', 'width': 160, 'height': 160})
                annotations.append({'id': i, 'image_id': i, 'category_id': i % 3,
                                    'bbox': [60, 60, 8, 10], 'area': 80, 'iscrowd': 0})
            annotation = root / 'fixture.json'
            annotation.write_text(json.dumps({'images': images, 'annotations': annotations,
                                             'categories': [{'id': i, 'name': str(i)} for i in range(3)]}))
            def fixture_config(output):
                cfg = config(DIRECT)
                cfg.device, cfg.output_dir = 'cpu', str(output)
                cfg.yaml_cfg['seed'] = 0
                for name in ('train_dataloader', 'val_dataloader'):
                    loader = cfg.yaml_cfg[name]
                    loader.update(num_workers=0, total_batch_size=2)
                    loader['dataset'].update(img_folder=str(root), ann_file=str(annotation))
                    for op in loader['dataset']['transforms']['ops']:
                        if op['type'] == 'Resize':
                            op['size'] = [160, 160]
                return cfg
            cfg = fixture_config(root / 'train')
            cfg.epoches = 1  # One synthetic epoch, not an experiment-config change.
            with contextlib.redirect_stdout(io.StringIO()):
                solver = DetSolver(cfg)
                solver.fit()
            if solver.writer:
                solver.writer.close()
            report = json.loads((Path(cfg.output_dir) / 'umqr_direct_alpha_final.json').read_text())
            state = torch.load(Path(cfg.output_dir) / 'last.pth', weights_only=False)
            self.assertEqual(report['alpha_model'], float(state['model']['decoder.umqr_head.alpha']))
            self.assertEqual(report['alpha_ema'], float(state['ema']['module']['decoder.umqr_head.alpha']))
            row = json.loads((Path(cfg.output_dir) / 'log.txt').read_text().splitlines()[-1])
            self.assertEqual(row['umqr_alpha_model'], report['alpha_model'])
            self.assertEqual(row['umqr_alpha_ema'], report['alpha_ema'])
            eval_cfg = fixture_config(root / 'evaluation')
            eval_cfg.resume = str(Path(cfg.output_dir) / 'best.pth')
            best = torch.load(eval_cfg.resume, weights_only=False)
            log = io.StringIO()
            with contextlib.redirect_stdout(log):
                evaluation = DetSolver(eval_cfg)
                evaluation.val()
            if evaluation.writer:
                evaluation.writer.close()
            focused = json.loads((Path(eval_cfg.output_dir) / 'umqr_direct_metrics.json').read_text())
            coco = json.loads((Path(eval_cfg.output_dir) / 'evaluation_metrics.json').read_text())
            for key in ('AP', 'AP50', 'AP75', 'APs', 'APm', 'AR100', 'AR75'):
                self.assertEqual(focused[key], coco[key])
            self.assertEqual(focused['alpha'], float(best['ema']['module']['decoder.umqr_head.alpha']))
            self.assertEqual(focused['alpha_model'], float(best['model']['decoder.umqr_head.alpha']))
            self.assertEqual(focused['weights'], 'ema')
            self.assertIn('Encoder Top-K Query Diagnosis', log.getvalue())
            self.assertTrue((Path(eval_cfg.output_dir) / 'eval.pth').exists())

    def test_two_process_full_model_ddp(self):
        if not torch.distributed.is_gloo_available():
            self.skipTest('Gloo unavailable')
        with socket.socket() as socket_:
            socket_.bind(('127.0.0.1', 0))
            port = socket_.getsockname()[1]
        torch.multiprocessing.spawn(distributed_worker, args=(port,), nprocs=2, join=True)


if __name__ == '__main__':
    unittest.main(verbosity=2)
