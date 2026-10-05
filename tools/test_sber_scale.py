"""E4' checks: one fixed reference-area bypass, no real SAR AP claim."""

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
from tools.test_sber import E4, analytic_maps
from src.core.yaml_utils import load_config
from src.solver.det_engine import train_one_epoch, check_sber_first_batch
from src.solver.det_solver import DetSolver
from src.zoo.rtdetr.sber import (
    SBER_AREA_THRESHOLD, SBERHead, sber_scale_mask, sber_scale_debug, apply_boundary_residual)
from src.zoo.rtdetr.utils import inverse_sigmoid

SCALE = ROOT / 'configs/rtdetrv2/rtdetrv2_r18vd_80e_ogsod_sber_scale_gbs64_gpu2_zxy.yml'


def all_large_reference(module, args):
    # Test fixture only: preserve DDP model parameters, alter rank-local inputs.
    refs = args[1].clone()
    refs[..., 2:] = inverse_sigmoid(torch.full_like(refs[..., 2:], .3))
    return (args[0], refs, *args[2:])


def distributed_worker(rank, port):
    torch.set_num_threads(1)
    torch.distributed.init_process_group(
        'gloo', init_method=f'tcp://127.0.0.1:{port}?use_libuv=0', rank=rank, world_size=2)
    try:
        cfg = config(SCALE)
        model = nn.parallel.DistributedDataParallel(cfg.model)
        if rank == 1:
            model.module.decoder.decoder.register_forward_pre_hook(all_large_reference)
        images, targets = torch.rand(2, 3, 160, 160), training_targets()
        if rank == 1:
            targets[1] = {'labels': torch.tensor([1]), 'boxes': torch.tensor([[.6, .5, .1, .1]])}
        for step in range(2):
            cfg.optimizer.zero_grad()
            outputs = model(images, targets)
            losses = cfg.criterion(outputs, targets)
            sum(losses.values()).backward()
            check_sber_first_batch(model, outputs, losses, step)
            assert all(p.grad is not None and torch.isfinite(p.grad).all()
                       for p in model.module.decoder.sber_head.parameters())
            if rank == 1 and step == 0:
                assert all(float(detail['large_query_ratio']) == 1 for detail in outputs['sber_debug'])
            cfg.optimizer.step()
    finally:
        torch.distributed.destroy_process_group()


class SBERScaleChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_cutoff_reference_source_exact_residual_and_gradients(self):
        refs = torch.tensor([[[.5, .5, .04, .04], [.5, .5, .08, .08], [.5, .5, .15, .15],
                              [.5, .5, .15001, .15], [.5, .5, .5, .5]]])
        mask = sber_scale_mask(refs)
        self.assertEqual(mask.flatten().tolist(), [True, True, True, False, False])
        self.assertEqual(SBER_AREA_THRESHOLD, .0225)
        # Detector sizes disagree with reference sizes; decision must use refs.
        det = refs.clone()
        det[0, 0, 2:] = .5
        det[0, 3] = torch.tensor([.99, .99, .05, .05])
        det.requires_grad_()
        raw = torch.tensor([[[.04, -.02, -.03, .01]]]).repeat(1, 5, 1).requires_grad_()
        masked = raw * mask
        original = apply_boundary_residual(det, raw)
        refined = apply_boundary_residual(det, masked, use_sber=mask)
        enabled, large = mask.squeeze(-1), ~mask.squeeze(-1)
        torch.testing.assert_close(refined[enabled], original[enabled], rtol=0, atol=0)
        torch.testing.assert_close(refined[large], det[large], rtol=0, atol=0)
        self.assertEqual(float((refined - det)[large].abs().max()), 0.)
        self.assertEqual(float(masked[large].abs().max()), 0.)
        # Merely zeroing edge offsets does not exactly bypass E4's projection.
        zero_only = apply_boundary_residual(det, masked)
        self.assertGreater(float((zero_only[large] - det[large]).abs().max()), 0.)
        refined.sum().backward()
        torch.testing.assert_close(det.grad[large], torch.ones_like(det.grad[large]), rtol=0, atol=0)
        torch.testing.assert_close(raw.grad[large], torch.zeros_like(raw.grad[large]), rtol=0, atol=0)
        self.assertGreater(float(raw.grad[enabled].abs().sum()), 0.)
        self.assertTrue(torch.isfinite(refined).all())
        details = sber_scale_debug(refs, mask, raw, masked, det, refined, 5)
        self.assertEqual(int(details['small_query_count']), 1)
        self.assertEqual(int(details['medium_query_count']), 2)
        self.assertEqual(int(details['large_query_count']), 2)
        for name, expected in [('small', 1), ('medium', 1), ('large', 0)]:
            self.assertEqual(float(details[f'{name}_sber_enabled_ratio']), expected)
        # Leading DN queries must not enter regular-query proportions.
        dn_refs = torch.cat((refs[:, -1:], refs), 1)
        dn_det = torch.cat((det.detach()[:, -1:], det.detach()), 1)
        dn_raw = torch.cat((raw.detach()[:, -1:], raw.detach()), 1)
        dn_mask = sber_scale_mask(dn_refs)
        dn_masked = dn_raw * dn_mask
        dn_refined = apply_boundary_residual(dn_det, dn_masked, use_sber=dn_mask)
        dn_details = sber_scale_debug(dn_refs, dn_mask, dn_raw, dn_masked, dn_det, dn_refined, 5)
        self.assertEqual(int(dn_details['large_query_count']), 2)
        # All-large retains the graph with zero finite head gradients; no DDP
        # unused-parameter policy change is necessary even on an all-large rank.
        head = SBERHead(2, 8)
        large_refs = refs[:, -1:]
        with torch.autocast('cpu', dtype=torch.bfloat16):
            fractions, _ = head(analytic_maps(), large_refs)
            prediction = apply_boundary_residual(large_refs, fractions * sber_scale_mask(large_refs),
                                                 use_sber=sber_scale_mask(large_refs))
        prediction.sum().backward()
        torch.testing.assert_close(prediction, large_refs, rtol=0, atol=0)
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all()
                            and p.grad.abs().sum() == 0 for p in head.parameters()))

    def test_config_init_and_full_decoder_branch_equivalence(self):
        e4, scale = load_config(str(E4), cfg={}), load_config(str(SCALE), cfg={})
        expected = copy.deepcopy(e4)
        expected['__include__'] = ['./rtdetrv2_r18vd_80e_ogsod_sber_gbs64_gpu2_zxy.yml']
        expected['output_dir'] = scale['output_dir']
        expected['RTDETRTransformerv2']['sber_scale_aware'] = True
        self.assertEqual(expected, scale)
        torch.manual_seed(0)
        e4_model = config(E4).model
        rng = torch.get_rng_state()
        torch.manual_seed(0)
        scale_cfg = config(SCALE)
        model = scale_cfg.model
        torch.testing.assert_close(rng, torch.get_rng_state(), rtol=0, atol=0)
        self.assertEqual(set(e4_model.state_dict()), set(model.state_dict()))
        for name, value in e4_model.state_dict().items():
            torch.testing.assert_close(value, model.state_dict()[name], rtol=0, atol=0)
        self.assertEqual(sum(p.numel() for p in model.decoder.sber_head.parameters()), 131461)
        images = torch.rand(2, 3, 160, 160)
        # Nonzero E4 offsets, all-small refs: complete E4' outputs match E4.
        with torch.no_grad():
            for bbox in e4_model.decoder.dec_bbox_head:
                bbox.layers[-1].bias.copy_(torch.tensor([.01, -.01, .1, .1]))
            e4_model.decoder.enc_bbox_head.layers[-1].bias[2:].fill_(-3.)
            e4_model.decoder.sber_head.offset_head.bias.copy_(torch.tensor([.2, -.1, .1, -.2]))
            nn.init.normal_(e4_model.decoder.sber_head.offset_head.weight, std=.002)
            model.load_state_dict(e4_model.state_dict(), strict=True)
        e4_model.eval()
        model.eval()
        masks = []
        hook = model.decoder.sber_head.register_forward_pre_hook(lambda module, args: masks.append(sber_scale_mask(args[1])))
        with torch.no_grad():
            original, predicted = e4_model(images), model(images)
        hook.remove()
        self.assertTrue(all(mask.all() for mask in masks))
        for key in original:
            torch.testing.assert_close(original[key], predicted[key], rtol=0, atol=0)

        # All-large refs with nonzero bbox and SBER heads: exact E0 trajectory.
        baseline = config(E0).model
        with torch.no_grad():
            common = baseline.state_dict()
            common.update({name: value for name, value in model.state_dict().items() if name in common})
            baseline.load_state_dict(common, strict=True)
            baseline.decoder.enc_bbox_head.layers[-1].bias[2:].fill_(2.)
            state = model.state_dict()
            state.update(baseline.state_dict())
            model.load_state_dict(state, strict=True)
        masks.clear()
        hook = model.decoder.sber_head.register_forward_pre_hook(lambda module, args: masks.append(sber_scale_mask(args[1])))
        baseline.eval()
        with torch.no_grad():
            original, predicted = baseline(images), model(images)
        hook.remove()
        self.assertTrue(all((~mask).all() for mask in masks))
        for key in original:
            torch.testing.assert_close(original[key], predicted[key], rtol=0, atol=0)

    def test_training_debug_checkpoint_ema_and_nine_eval_metrics(self):
        cfg = config(SCALE)
        model, optimizer, criterion = cfg.model, cfg.optimizer, cfg.criterion
        images, targets = torch.rand(2, 3, 160, 160), training_targets()
        log = io.StringIO()
        with contextlib.redirect_stdout(log):
            stats = train_one_epoch(model, criterion, [(images, targets)] * 3,
                                    optimizer, torch.device('cpu'), 0, max_norm=.1)
        report = log.getvalue()
        self.assertEqual(report.count('SBER_FIRST_BATCH'), 1)
        self.assertIn("'large_boundary_residual_abs_max': 0.0", report)
        self.assertIn("'large_bbox_exact_baseline': True", report)
        self.assertIn("'small_medium_fraction_exact_e4': True", report)
        self.assertIn("'all_ranks_passed': True", report)
        self.assertTrue(all(torch.isfinite(torch.tensor(v)) for v in stats.values()))
        self.assertEqual(criterion.weight_dict, {'loss_vfl': 1, 'loss_bbox': 5, 'loss_giou': 2})
        model.eval()
        ema = cfg.ema
        ema.update(model)
        from PIL import Image
        with tempfile.TemporaryDirectory(prefix='sber_scale_') as directory:
            root = Path(directory)
            checkpoint = root / 'best.pth'
            torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                        'ema': ema.state_dict(), 'criterion': criterion.state_dict(),
                        'last_epoch': 0}, checkpoint)
            state = torch.load(checkpoint, weights_only=True)
            recovered = config(SCALE)
            recovered.model.load_state_dict(state['model'], strict=True)
            recovered.optimizer.load_state_dict(state['optimizer'])
            recovered.ema.load_state_dict(state['ema'])
            self.assertIn('exp_avg', recovered.optimizer.state[recovered.model.decoder.sber_head.offset_head.weight])
            recovered.model.eval()
            with torch.no_grad():
                torch.testing.assert_close(model(images)['pred_boxes'], recovered.model(images)['pred_boxes'])
            annotation = root / 'fixture.json'
            records, annotations = [], []
            for i in range(3):
                Image.new('RGB', (160, 160), (80, 100, 120)).save(root / f'{i}.png')
                records.append({'id': i, 'file_name': f'{i}.png', 'width': 160, 'height': 160})
                annotations.append({'id': i, 'image_id': i, 'category_id': i,
                                    'bbox': [60, 60, 8, 10], 'area': 80, 'iscrowd': 0})
            annotation.write_text(json.dumps({'images': records, 'annotations': annotations,
                                             'categories': [{'id': i, 'name': str(i)} for i in range(3)]}))
            evaluation_cfg = config(SCALE)
            evaluation_cfg.device, evaluation_cfg.output_dir = 'cpu', str(root / 'eval_best')
            evaluation_cfg.resume = str(checkpoint)
            evaluation_cfg.yaml_cfg['seed'] = 0
            loader = evaluation_cfg.yaml_cfg['val_dataloader']
            loader.update(num_workers=0, total_batch_size=2)
            loader['dataset'].update(img_folder=str(root), ann_file=str(annotation))
            for op in loader['dataset']['transforms']['ops']:
                if op['type'] == 'Resize':
                    op['size'] = [160, 160]
            with contextlib.redirect_stdout(io.StringIO()) as output:
                solver = DetSolver(evaluation_cfg)
                solver.val()
            if solver.writer:
                solver.writer.close()
            result = Path(evaluation_cfg.output_dir)
            focused = json.loads((result / 'sber_scale_metrics.json').read_text())
            full = json.loads((result / 'evaluation_metrics.json').read_text())
            for name in ('AP', 'AP50', 'AP75', 'APs', 'APm', 'APl', 'AR100', 'ARs', 'AR75'):
                self.assertEqual(focused[name], full[name])
            self.assertEqual(focused['weights'], 'ema')
            self.assertIn('0.0225', focused['sber_bypass'])
            self.assertIn('Encoder Top-K Query Diagnosis', output.getvalue())
            self.assertTrue((result / 'eval.pth').exists())

    def test_two_process_ddp_with_all_large_rank(self):
        if not torch.distributed.is_gloo_available():
            self.skipTest('Gloo unavailable')
        with socket.socket() as socket_:
            socket_.bind(('127.0.0.1', 0))
            port = socket_.getsockname()[1]
        torch.multiprocessing.spawn(distributed_worker, args=(port,), nprocs=2, join=True)


if __name__ == '__main__':
    unittest.main(verbosity=2)
