"""CPU synthetic checks for E2; no real dataset or pretrained download."""

import copy
import contextlib
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
from src.core.yaml_utils import load_config
from src.optim import ModelEMA
from src.solver.det_solver import DetSolver
from src.solver.query_selection_metrics import QuerySelectionMetrics
from src.zoo.rtdetr.sar_quality import SARQualityHead


CONFIG_DIR = Path(__file__).resolve().parents[1] / 'configs/rtdetrv2'
E1 = CONFIG_DIR / 'rtdetrv2_r18vd_80e_ogsod_qaqs_gbs64_gpu2_zxy.yml'
E2 = CONFIG_DIR / 'rtdetrv2_r18vd_80e_ogsod_sqqs_gbs64_gpu2_zxy.yml'


def make_config(path):
    cfg = YAMLConfig(str(path))
    cfg.yaml_cfg['PResNet']['pretrained'] = False
    # Synthetic size only; keep all 300 queries, 3 layers and ordinary DN.
    cfg.yaml_cfg['eval_spatial_size'] = [160, 160]
    return cfg


def metric_fixture(image_id=1):
    # First small GT's best query is rank 2; second is missed (rank-1 tie).
    outputs = {'enc_topk_boxes': torch.tensor([
        [[.8, .8, .1, .1], [.15, .15, .1, .1], [.15, .15, .08, .08]]])}
    target = {
        'boxes': torch.tensor([[10., 10., 20., 20.], [40., 40., 50., 50.]]),
        'area': torch.tensor([100., 100.]),
        'orig_size': torch.tensor([100, 100]),
        'image_id': torch.tensor([image_id]),
    }
    return outputs, [target]


def distributed_worker(rank, port):
    torch.set_num_threads(1)
    torch.distributed.init_process_group(
        'gloo', init_method=f'tcp://127.0.0.1:{port}?use_libuv=0', rank=rank, world_size=2)
    try:
        stats = QuerySelectionMetrics()
        # A padded ID appears on both ranks; a unique image appears on rank 1.
        outputs, targets = metric_fixture()
        stats.update(outputs, targets, (100, 100))
        if rank == 1:
            outputs, targets = metric_fixture(2)
            stats.update(outputs, targets, (100, 100))
        summary = stats.summarize()
        assert summary['unique_image_count'] == 2
        assert summary['small_gt_count'] == 4
        assert summary['small_query_recall_iou75'] == .5
        assert summary['small_best_query_rank']['mean'] == 1.5

        head = nn.parallel.DistributedDataParallel(SARQualityHead(8))
        optimizer = torch.optim.AdamW(head.parameters(), lr=.001)
        for _ in range(2):
            optimizer.zero_grad()
            features = torch.randn(2, 16, 8)
            proposals = torch.randn(2, 16, 4)
            prediction = head(features, proposals, [(4, 4)])
            prediction.square().add(prediction).mean().backward()
            assert all(p.grad is not None for p in head.parameters())
            optimizer.step()
    finally:
        torch.distributed.destroy_process_group()


class SQQSChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_locality_and_geometry_gradient(self):
        head = SARQualityHead(2)
        constant = torch.cat((torch.ones(1, 6, 2), torch.full((1, 2, 2), 10.)), 1)
        residual = head.structural_residual(constant, [(2, 3), (1, 2)])
        torch.testing.assert_close(residual, torch.zeros_like(residual))
        features = constant.clone().requires_grad_()
        features.data[0, 1, 0] += 3
        residual = head.structural_residual(features, [(2, 3), (1, 2)])
        self.assertNotEqual(float(residual[0, 1, 0]), 0.)
        torch.testing.assert_close(residual[:, 6:], torch.zeros_like(residual[:, 6:]))
        proposals = torch.randn(1, 8, 4, requires_grad=True)
        nn.init.normal_(head.proj.weight)
        head(features, proposals, [(2, 3), (1, 2)]).sum().backward()
        self.assertIsNone(proposals.grad)
        self.assertGreater(float(features.grad.abs().sum()), 0.)
        for pool_size in (1, 2, 2.5):
            with self.assertRaises(ValueError):
                SARQualityHead(2, pool_size)

    def test_config_initialization_training_checkpoint(self):
        e1, e2 = load_config(str(E1)), load_config(str(E2))
        expected = copy.deepcopy(e1)
        expected['output_dir'] = e2['output_dir']
        expected['query_selection_metrics'] = True
        expected['RTDETRTransformerv2'].update(
            sar_quality=True, sar_quality_pool_size=3)
        self.assertEqual(expected, e2)

        torch.manual_seed(0)
        e1_model = make_config(E1).model
        e1_rng = torch.get_rng_state()
        torch.manual_seed(0)
        cfg = make_config(E2)
        model, criterion, optimizer = cfg.model, cfg.criterion, cfg.optimizer
        torch.testing.assert_close(e1_rng, torch.get_rng_state())
        for name, value in e1_model.state_dict().items():
            if not name.startswith('decoder.enc_quality_head.'):
                torch.testing.assert_close(value, model.state_dict()[name], rtol=0, atol=0)
        self.assertEqual(model.decoder.num_queries, 300)
        self.assertEqual(model.decoder.num_layers, 3)
        self.assertEqual(cfg.yaml_cfg['train_dataloader']['total_batch_size'], 64)
        quality_params = list(model.decoder.enc_quality_head.parameters())
        optimizer_ids = {id(p) for group in optimizer.param_groups for p in group['params']}
        self.assertTrue(all(id(p) in optimizer_ids for p in quality_params))

        images = torch.rand(2, 3, 160, 160)
        model.eval()
        e1_model.eval()
        with torch.no_grad():
            before, baseline = model(images), e1_model(images)
        for name in ('pred_logits', 'pred_boxes', 'enc_topk_indices'):
            torch.testing.assert_close(before[name], baseline[name], rtol=0, atol=0)
        torch.testing.assert_close(
            before['enc_topk_quality_logits'], torch.zeros(2, 300, 1))

        targets = [
            {'labels': torch.tensor([0]), 'boxes': torch.tensor([[.3, .4, .06, .08]])},
            {'labels': torch.empty(0, dtype=torch.long), 'boxes': torch.empty(0, 4)},
        ]
        ema = ModelEMA(model)
        model.train()
        for _ in range(2):
            optimizer.zero_grad()
            outputs = model(images, targets)
            losses = criterion(outputs, targets)
            self.assertIn('loss_quality', losses)
            self.assertIn('dn_aux_outputs', outputs)
            self.assertTrue(all(torch.isfinite(loss) for loss in losses.values()))
            sum(losses.values()).backward()
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all()
                                for p in quality_params))
            # Feature, residual and proposal-geometry columns all receive gradients.
            gradient = model.decoder.enc_quality_head.proj.weight.grad
            for block in (gradient[:, :256], gradient[:, 256:512], gradient[:, 512:]):
                self.assertGreater(float(block.abs().sum()), 0.)
            optimizer.step()
            ema.update(model)

        # Existing supervision alone does not backpropagate into the bbox head.
        optimizer.zero_grad()
        outputs = model(images, targets)
        criterion.loss_quality(outputs, targets)['loss_quality'].backward()
        self.assertTrue(all(p.grad is None for p in model.decoder.enc_bbox_head.parameters()))

        stream = io.BytesIO()
        torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                    'ema': ema.state_dict(), 'criterion': criterion.state_dict()}, stream)
        stream.seek(0)
        state = torch.load(stream, weights_only=True)
        restored_cfg = make_config(E2)
        restored = restored_cfg.model
        restored.load_state_dict(state['model'], strict=True)
        restored_cfg.optimizer.load_state_dict(state['optimizer'])
        restored_cfg.criterion.load_state_dict(state['criterion'], strict=True)
        restored_ema = ModelEMA(restored)
        restored_ema.load_state_dict(state['ema'], strict=True)
        for parameter in restored.decoder.enc_quality_head.parameters():
            self.assertIn('exp_avg', restored_cfg.optimizer.state[parameter])
        model.eval()
        restored.eval()
        with torch.no_grad():
            torch.testing.assert_close(model(images)['pred_boxes'], restored(images)['pred_boxes'])
        torch.testing.assert_close(ema.module.decoder.enc_quality_head.proj.weight,
                                   restored_ema.module.decoder.enc_quality_head.proj.weight)

        decoder = model.decoder
        memory = torch.rand(1, 3, 256)
        logits = torch.logit(torch.tensor([[[.9], [.8], [.7]]])).expand(-1, -1, 3)
        quality = torch.logit(torch.tensor([[[.1], [.9], [.8]]]))
        coords = torch.zeros(1, 3, 4)
        selected = decoder._select_topk(memory, logits, coords, 2, quality)[3]
        self.assertEqual(selected.tolist(), [[1, 2]])
        decoder.quality_beta = 0.
        selected = decoder._select_topk(memory, logits, coords, 2, quality)[3]
        self.assertEqual(selected.tolist(), [[0, 1]])

    def test_metric_definition_empty_and_dedup(self):
        stats = QuerySelectionMetrics()
        outputs, targets = metric_fixture()
        stats.update(outputs, targets, (100, 100))
        stats.update(outputs, targets, (100, 100))
        result = stats.summarize()
        self.assertEqual(result['small_query_recall_iou75'], .5)
        self.assertEqual(result['small_best_query_rank'], {'mean': 1.5, 'median': 1.5, 'P90': 1.9})
        self.assertEqual(result['unique_image_count'], 1)
        merged = stats.merge_image_records([stats.images, stats.images])
        self.assertEqual(len(merged), 1)
        empty = QuerySelectionMetrics().summarize()
        self.assertIsNone(empty['small_query_recall_iou75'])
        self.assertIsNone(empty['small_best_query_rank']['P90'])

    def test_two_process_cpu_ddp(self):
        if not torch.distributed.is_gloo_available():
            self.skipTest('Gloo unavailable')
        with socket.socket() as socket_:
            socket_.bind(('127.0.0.1', 0))
            port = socket_.getsockname()[1]
        torch.multiprocessing.spawn(distributed_worker, args=(port,), nprocs=2, join=True)

    def test_full_evaluation_outputs_and_legacy_diagnosis(self):
        from PIL import Image
        with tempfile.TemporaryDirectory(prefix='sqqs_eval_') as directory:
            root = Path(directory)
            images, annotations = [], []
            for image_id in range(2):
                Image.new('RGB', (160, 160), (80, 100, 120)).save(root / f'{image_id}.png')
                images.append({'id': image_id, 'file_name': f'{image_id}.png',
                               'width': 160, 'height': 160})
                annotations.append({'id': image_id, 'image_id': image_id,
                                    'category_id': image_id, 'bbox': [30, 40, 8, 10],
                                    'area': 80, 'iscrowd': 0})
            annotation = root / 'val.json'
            annotation.write_text(json.dumps({
                'images': images, 'annotations': annotations,
                'categories': [{'id': i, 'name': f'class_{i}'} for i in range(3)]}),
                encoding='utf-8')
            cfg = make_config(E2)
            cfg.device = 'cpu'
            cfg.seed = 0
            cfg.output_dir = str(root / 'evaluation')
            cfg.yaml_cfg['seed'] = 0
            loader = cfg.yaml_cfg['val_dataloader']
            loader['num_workers'] = 0
            loader['total_batch_size'] = 2
            loader['dataset']['img_folder'] = str(root)
            loader['dataset']['ann_file'] = str(annotation)
            for operation in loader['dataset']['transforms']['ops']:
                if operation['type'] == 'Resize':
                    operation['size'] = [160, 160]
            checkpoint = root / 'synthetic.pth'
            torch.save({'model': cfg.model.state_dict(), 'ema': cfg.ema.state_dict(),
                        'criterion': cfg.criterion.state_dict(), 'last_epoch': 0}, checkpoint)
            cfg.resume = str(checkpoint)
            with contextlib.redirect_stdout(io.StringIO()):
                solver = DetSolver(cfg)
                solver.val()
            output = Path(cfg.output_dir)
            focused = json.loads((output / 'query_selection_metrics.json').read_text())
            coco = json.loads((output / 'evaluation_metrics.json').read_text())
            self.assertEqual(focused['unique_image_count'], 2)
            self.assertEqual(focused['small_gt_count'], 2)
            self.assertEqual(focused['num_queries'], [300])
            self.assertEqual(focused['checkpoint'], str(checkpoint))
            self.assertEqual(focused['weights'], 'ema')
            self.assertEqual(focused['config']['seed'], 0)
            for name in ('AP', 'AP50', 'AP75', 'APs'):
                self.assertEqual(focused[name], coco[name])
            self.assertTrue((output / 'eval.pth').exists())
            self.assertTrue((output / 'query_diagnosis').is_dir())
            if solver.writer:
                solver.writer.close()


if __name__ == '__main__':
    unittest.main(verbosity=2)
