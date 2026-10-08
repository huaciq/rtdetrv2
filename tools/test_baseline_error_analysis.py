"""Synthetic correctness checks. These are NOT OGSOD experiment results."""
import argparse
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
from PIL import Image

import baseline_error_analysis as analysis


class ErrorAnalysisTest(unittest.TestCase):
    def test_baseline_export_strict_ema_and_original_postprocessor(self):
        import torch
        from src.core import YAMLConfig
        torch.set_num_threads(1)
        baseline = analysis.ROOT / "configs/rtdetrv2/rtdetrv2_r18vd_80e_ogsod_baseline_gbs64_gpu2_zxy.yml"
        with tempfile.TemporaryDirectory(prefix="synthetic_export_") as temp:
            root = Path(temp)
            cfg_file = root / "config.yml"
            # Only initialization/download and loader worker count are changed in this fixture.
            cfg_file.write_text(f"__include__: ['{baseline.as_posix()}']\nPResNet:\n  pretrained: False\nval_dataloader:\n  num_workers: 0\n", encoding="utf-8")
            cfg = YAMLConfig(str(cfg_file))
            weights = cfg.model.state_dict()
            checkpoint = root / "synthetic.pth"
            torch.save({"ema": {"module": weights}, "model": weights, "last_epoch": 0}, checkpoint)
            images = []
            for i in (1, 2):
                Image.new("RGB", (200, 200), color=(80, 80, 80)).save(root / f"{i}.png")
                images.append(dict(id=i, file_name=f"{i}.png", width=200, height=200))
            annotation_file = root / "gt.json"
            analysis.dump(annotation_file, dict(images=images, annotations=[],
                                                categories=[dict(id=i, name=str(i)) for i in range(3)]))
            out = root / "out"
            out.mkdir()
            args = argparse.Namespace(config=str(cfg_file), checkpoint=str(checkpoint),
                                      annotations=str(annotation_file), images=str(root), output=str(out))
            with mock.patch("torch.cuda.is_available", return_value=False):
                predictions, metadata = analysis.export(args)
            self.assertEqual(metadata["weights"], "ema")
            self.assertTrue(metadata["strict_load"])
            self.assertEqual(len(predictions), 600)
            self.assertEqual(set(p["image_id"] for p in predictions), {1, 2})
            self.assertEqual(set(predictions[0]), {"image_id", "category_id", "score", "bbox"})
            self.assertTrue(all(np.isfinite(p["bbox"]).all() for p in predictions))
            self.assertFalse(cfg.model.decoder.umqr)
            self.assertFalse(cfg.model.decoder.sber)
            from faster_coco_eval.utils.pytorch import FasterCocoEvaluator
            with torch.inference_mode():
                sample = torch.full((2, 3, 640, 640), 80/255)
                original = cfg.postprocessor(cfg.model.eval()(sample), torch.tensor([[200, 200], [200, 200]]))
            from faster_coco_eval import COCO
            coco = COCO(str(annotation_file))
            evaluator = FasterCocoEvaluator(coco, ["bbox"])
            reference = evaluator.prepare_for_coco_detection(dict(zip((1, 2), original)))
            self.assertEqual(predictions, reference)

    def test_complete_fixed_analysis(self):
        with tempfile.TemporaryDirectory(prefix="synthetic_error_analysis_") as temp:
            root = Path(temp)
            images, annotations, predictions = [], [], []
            expected_miss50 = set()
            for i in range(1, 85):
                cat = (i - 1) % 3
                edge = (20, 50, 110)[(i - 1) % 3]
                box = [40, 40, edge, edge]
                images.append(dict(id=i, file_name=f"{i}.png", width=200, height=200))
                Image.new("RGB", (200, 200), color=(i, i, i)).save(root / f"{i}.png")
                annotations.append(dict(id=i, image_id=i, category_id=cat, bbox=box,
                                        area=edge**2, iscrowd=0))
                if i <= 24:  # TP + lower-scoring duplicate.
                    predictions += [dict(image_id=i, category_id=cat, bbox=box, score=.95),
                                    dict(image_id=i, category_id=cat, bbox=box, score=.7)]
                elif i <= 48:  # Far-away high-confidence background FP and missed GT.
                    predictions.append(dict(image_id=i, category_id=cat, bbox=[170, 170, 10, 10], score=.9))
                    expected_miss50.add(i)
                elif i <= 60:  # Localization errors, IoU ~.43.
                    predictions.append(dict(image_id=i, category_id=cat,
                                            bbox=[40+edge*.4, 40, edge, edge], score=.9))
                    expected_miss50.add(i)
                elif i <= 72:  # Classification errors.
                    predictions.append(dict(image_id=i, category_id=(cat+1) % 3, bbox=box, score=.9))
                    expected_miss50.add(i)
                else:  # Both class and localization errors.
                    predictions.append(dict(image_id=i, category_id=(cat+1) % 3,
                                            bbox=[40+edge*.4, 40, edge, edge], score=.9))
                    expected_miss50.add(i)
            gt = dict(images=images, annotations=annotations,
                      categories=[dict(id=i, name=f"class_{i}") for i in range(3)])
            annotation_file = root / "gt.json"
            analysis.dump(annotation_file, gt)
            out = root / "analysis"
            out.mkdir()
            args = argparse.Namespace(output=str(out), annotations=str(annotation_file), images=str(root))
            analysis.analyze(args, predictions, {"predictions_source": "SYNTHETIC_FIXTURE"})
            tide = json.loads((out / "tide_summary.json").read_text())
            self.assertTrue(all(tide["counts"][k] > 0 for k in analysis.ERRORS))
            metrics, matches, missed = analysis.coco_evaluate(out / "validation_gt.json", predictions, out)
            self.assertEqual(missed[.5], expected_miss50)
            self.assertEqual(matches[(0, .5)], (1, False))
            self.assertEqual(matches[(1, .5)], (0, False))
            self.assertAlmostEqual(metrics["AR100"], 24/84, places=6)
            from tidecv import TIDE
            from tidecv.data import Data
            truth, detected = Data("fixture"), Data("prediction")
            for c in gt["categories"]:
                truth.add_class(c["id"], c["name"])
            for a in annotations:
                truth.add_ground_truth(a["image_id"], a["category_id"], box=a["bbox"])
            for p in predictions:
                detected.add_detection(p["image_id"], p["category_id"], p["score"], box=p["bbox"])
            official = TIDE()
            official.evaluate(truth, detected, name="fixture")
            self.assertEqual(tide["dAP_percent"], official.get_main_errors()["fixture"])
            complete = json.loads((out / "COMPLETE.json").read_text())
            self.assertEqual(complete["visualizations"], {"TP": 20, "FP": 20, "FN": 20})
            self.assertEqual(len(list((out / "visualizations").rglob("*.png"))), 120)
            self.assertTrue((out / "ERROR_ANALYSIS.md").is_file())
            self.assertTrue((out / "error_summary.csv").is_file())

    def test_empty_predictions_crowd_and_iou75(self):
        with tempfile.TemporaryDirectory(prefix="synthetic_error_analysis_") as temp:
            out = Path(temp)
            gt = dict(images=[dict(id=1, file_name="1.png", width=200, height=200)],
                      categories=[dict(id=0, name="one")],
                      annotations=[dict(id=1, image_id=1, category_id=0, bbox=[20, 20, 20, 20], area=400, iscrowd=0),
                                   dict(id=2, image_id=1, category_id=0, bbox=[100, 100, 60, 60], area=3600, iscrowd=1)])
            analysis.dump(out / "gt.json", gt)
            p = [dict(image_id=1, category_id=0, bbox=[25, 20, 20, 20], score=.9),
                 dict(image_id=1, category_id=0, bbox=[110, 110, 10, 10], score=.9)]
            _, matches, missed = analysis.coco_evaluate(out / "gt.json", p, out)
            self.assertEqual(matches[(0, .5)], (1, False))
            self.assertEqual(matches[(1, .5)][1], True)
            self.assertEqual(missed[.5], set())
            self.assertEqual(missed[.75], {1})
            metrics, _, missed = analysis.coco_evaluate(out / "gt.json", [], out)
            self.assertEqual(metrics["AP"], 0)
            self.assertEqual(missed[.5], {1})
            tide = analysis.tide_evaluate(gt, [], out)
            self.assertEqual(tide["counts"]["Miss"], 1)
            self.assertIsNone(tide["dAP_percent"]["Miss"])

    def test_coverage_is_not_unique_recall(self):
        # One prediction can geometrically cover two boxes; COCO only matches one.
        p = [dict(bbox=[10, 10, 20, 20])]
        g = [dict(bbox=[10, 10, 20, 20]), dict(bbox=[11, 10, 20, 20])]
        self.assertTrue((analysis.ious(p, g) >= .75).all())
        np.testing.assert_array_equal(analysis.ious([], g).shape, (0, 2))
        self.assertEqual(analysis.size(32**2), "medium")
        self.assertEqual(analysis.size(96**2), "large")

    def test_localization_size_uses_same_class_gt(self):
        with tempfile.TemporaryDirectory() as temp:
            gt = dict(images=[dict(id=1)], annotations=[
                dict(id=1, image_id=1, category_id=0, bbox=[0, 0, 50, 50], area=2500),
                dict(id=2, image_id=1, category_id=1, bbox=[0, 0, 20, 20], area=400)])
            p = [dict(image_id=1, category_id=0, bbox=[0, 0, 20, 20], score=.9)]
            rows, _ = analysis.details(gt, p, {(0, .5): (0, False)}, {.5: {1, 2}, .75: {1, 2}}, Path(temp))
            self.assertEqual(rows[0]["error50"], "Loc")
            self.assertEqual(rows[0]["target_size"], "medium")
            self.assertEqual(rows[0]["nearest_gt_id"], 2)
            self.assertEqual(rows[0]["error_target_gt_id"], 1)


if __name__ == "__main__":
    unittest.main()
