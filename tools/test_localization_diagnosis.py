"""Focused correctness checks using synthetic geometry/COCO; no SAR claims."""
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
from PIL import Image
import yaml

import localization_diagnosis as diag
import localization_pilot_check as pilot_check
import prepare_dfine_pilot as prep


def fixture():
    return {"info": {}, "images": [{"id": 1, "file_name": "val.png", "width": 640, "height": 640}],
            "categories": [{"id": 0, "name": "bridge"}, {"id": 1, "name": "harbor"}, {"id": 2, "name": "storage_tank"}],
            "annotations": [{"id": 1, "image_id": 1, "category_id": 0, "bbox": [20, 20, 6, 6], "area": 36, "iscrowd": 0},
                            {"id": 2, "image_id": 1, "category_id": 0, "bbox": [100, 100, 40, 40], "area": 1600, "iscrowd": 0}]}


class GeometryTests(unittest.TestCase):
    def test_axis_oracles(self):
        gt = np.array([10., 10., 30., 30.])
        for predicted, axis in (([13, 10, 33, 30], "center"), ([8, 8, 32, 32], "area_scale"),
                                ([7.5, 12, 32.5, 28], "aspect_ratio")):
            error = diag.geometry_error(np.array(predicted, float), gt)
            self.assertEqual(error["dominant_oracle"], axis)
            self.assertAlmostEqual(error["iou_fix_"+axis], 1.)
        perfect = diag.geometry_error(gt, gt)
        self.assertEqual(perfect["dominant_oracle"], "mixed_or_tied")
        self.assertEqual(perfect["mean_abs_boundary_px"], 0.)

    def test_resize_clipping_and_bin_boundaries(self):
        image = {"width": 1280, "height": 320}
        edges = diag.input_box([-4, 2, 20, 8], image, [640, 640], clip=True)
        np.testing.assert_allclose(edges, [0, 4, 8, 20])
        self.assertEqual([diag.short_bin(x) for x in (7.99, 8, 16, 32)], list(diag.BINS))
        self.assertEqual(diag.canonical_category("Storage_Tank"), "Storage Tank")
        with self.assertRaises(ValueError):
            diag.canonical_category("unknown")

    def test_bin_ignores_other_gt_and_unmatched_other_size_predictions(self):
        gt = fixture()
        predictions = [{"image_id": 1, "category_id": 0, "bbox": [100, 100, 40, 40], "score": .99},
                       {"image_id": 1, "category_id": 0, "bbox": [300, 300, 40, 40], "score": .95},
                       {"image_id": 1, "category_id": 0, "bbox": [20, 20, 6, 6], "score": .9}]
        metrics, _ = diag.group_metrics(gt, predictions, {1: gt["images"][0]}, [640, 640], "<8")
        self.assertAlmostEqual(metrics[0]["AP75"], 1.)
        self.assertEqual(metrics[0]["gt_count"], 1)
        self.assertIsNone(metrics[1]["AP75"])
        all_metrics, _ = diag.group_metrics(gt, predictions, {1: gt["images"][0]}, [640, 640])
        self.assertLess(all_metrics[0]["AP75"], 1.)

    def test_one_to_one_matching_and_no_fake_fn_geometry(self):
        gt = fixture()
        gt["annotations"][0]["bbox"] = [10, 10, 20, 20]
        gt["annotations"][0]["area"] = 400
        predictions = [{"image_id": 1, "category_id": 0, "bbox": [15, 10, 20, 20], "score": .9}]
        images = {1: gt["images"][0]}
        _, evaluator = diag.group_metrics(gt, predictions, images, [640, 640])
        rows, errors = diag.geometry_rows(gt, predictions, images, {0: "Bridge", 1: "Harbor", 2: "Storage Tank"}, [640, 640], evaluator)
        self.assertEqual(len(errors), 1)
        self.assertAlmostEqual(errors[0]["iou"], .6)
        self.assertEqual(errors[0]["dominant_oracle"], "center")
        self.assertTrue(all(r["miss75"] for r in rows))
        self.assertIsNone(rows[1]["matched50_iou"])
        m = pilot_check.metrics(gt, predictions)
        self.assertEqual(m["class_AP75"]["Bridge"], 0.)
        self.assertIsNone(m["class_AP75"]["Storage Tank"])
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            saved = {k: m[k] for k in (*diag.base.METRICS, "AR75")}
            diag.base.dump(root/"coco_metrics.json", saved)
            diag.base.write_csv(root/"error_summary.csv", [
                {"section": "COCO", "metric": k, "value": v*100 if v >= 0 else ""} for k, v in saved.items()])
            self.assertTrue(diag.verify_existing_coco(root, evaluator)["saved_COCO_recomputed"])
            saved["AP75"] = .4
            diag.base.dump(root/"coco_metrics.json", saved)
            with self.assertRaisesRegex(ValueError, "Saved COCO"):
                diag.verify_existing_coco(root, evaluator)

    def test_artifact_render_and_manual_review_pending(self):
        gt = fixture()
        predictions = [{"image_id": 1, "category_id": 0, "bbox": [21, 20, 6, 6], "score": .9}]
        images, cats = {1: gt["images"][0]}, {0: "Bridge", 1: "Harbor", 2: "Storage Tank"}
        metrics, evaluator = diag.group_metrics(gt, predictions, images, [640, 640])
        rows, errors = diag.geometry_rows(gt, predictions, images, cats, [640, 640], evaluator)
        for group in diag.BINS:
            extra, _ = diag.group_metrics(gt, predictions, images, [640, 640], group)
            metrics += extra
        summary = diag.summarize(rows, errors, metrics, cats)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            Image.new("RGB", (640, 640)).save(root/"val.png")
            diag.charts(rows, errors, summary, root)
            diag.visuals(rows, errors, gt, predictions, images, root, root, root)
            self.assertEqual(len(list(root.glob("*_size_iou.png"))), 3)
            self.assertIn("pending", (root/"manual_review.csv").read_text(encoding="utf-8-sig"))
            self.assertEqual(len(list((root/"examples").glob("*.png"))), 2)  # original and crop

    def test_provenance_accepts_byte_link_and_rejects_wrong_checkpoint(self):
        import torch
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            src, analyzed = root/"source", root/"analysis"
            src.mkdir(); analyzed.mkdir()
            gt = fixture()
            train = copy.deepcopy(gt)
            train["images"][0]["file_name"] = "train.png"
            for name, value in (("val.json", gt), ("train.json", train)):
                diag.base.dump(root/name, value)
            Image.new("RGB", (640, 640)).save(root/"val.png")
            cfg = {"eval_spatial_size": [640, 640], "PResNet": {"pretrained": True},
                   "train_dataloader": {"dataset": {"ann_file": str(root/"train.json")}},
                   "val_dataloader": {"dataset": {"ann_file": str(root/"val.json"), "img_folder": str(root),
                     "transforms": {"ops": [{"type": "Resize", "size": [640, 640]},
                                             {"type": "ConvertPILImage", "dtype": "float32", "scale": True}]}}},
                   "RTDETRPostProcessor": {"num_top_queries": 3}}
            (root/"config.yml").write_text(yaml.safe_dump(cfg), encoding="utf-8")
            torch.save({"last_epoch": 5, "ema": {"module": {"backbone.weight": torch.ones(1)}}}, root/"best.pth")
            recorded = copy.deepcopy(cfg); recorded["PResNet"]["pretrained"] = False
            diag.base.dump(src/"export_metadata.json", {"checkpoint": str(root/"best.pth"),
                "checkpoint_sha256": diag.base.sha(root/"best.pth"), "last_epoch": 5, "weights": "ema", "strict_load": True,
                "resolved_config": recorded, "image_count": 1, "gpu_process_count": 2})
            predictions = [{"image_id": 1, "category_id": i, "bbox": [20, 20, 6, 6], "score": .8} for i in range(3)]
            for directory in (src, analyzed):
                diag.base.dump(directory/"predictions.json", predictions)
            diag.base.dump(analyzed/"validation_gt.json", gt)
            diag.base.dump(analyzed/"analysis_manifest.json", {"predictions_sha256": diag.base.sha(analyzed/"predictions.json"),
                "annotations_source": str(root/"val.json"), "annotations_sha256": diag.base.sha(root/"val.json"), "git_commit": "0"*40})
            diag.base.dump(analyzed/"COMPLETE.json", {"status": "complete"})
            (analyzed/"ERROR_ANALYSIS.md").touch(); (analyzed/"error_summary.csv").touch()
            args = SimpleNamespace(export_dir=str(src), analysis_dir=str(analyzed), checkpoint=str(root/"best.pth"),
                                   config=str(root/"config.yml"), train_annotations=str(root/"train.json"), images=str(root))
            result = diag.verify_provenance(args)[-1]
            self.assertIsNone(result["export_git_commit"])
            self.assertTrue(all(c["passed"] for c in result["checks"]))
            with (root/"best.pth").open("ab") as stream:
                stream.write(b"changed")
            with self.assertRaisesRegex(ValueError, "checkpoint_bytes"):
                diag.verify_provenance(args)

    def test_pilot_config_only_truncates_baseline_and_protocol_detects_mismatch(self):
        from src.core import YAMLConfig
        baseline = diag.resolved_config(diag.base.ROOT/"configs/rtdetrv2/rtdetrv2_r18vd_80e_ogsod_baseline_gbs64_gpu2_zxy.yml")
        pilot = diag.resolved_config(diag.base.ROOT/"configs/diagnosis/rtdetrv2_r18_ogsod_pilot20.yml")
        native = YAMLConfig(str(diag.base.ROOT/"configs/diagnosis/rtdetrv2_r18_ogsod_pilot20.yml")).yaml_cfg
        native.pop("__include__", None)
        self.assertEqual(native, pilot)
        changed = {k for k in baseline.keys() | pilot.keys() if baseline.get(k) != pilot.get(k)}
        self.assertEqual(changed, {"epoches", "checkpoint_freq", "output_dir"})
        cfg = {k: copy.deepcopy(pilot[k]) for k in prep.SHARED}
        cfg["train_dataloader"]["collate_fn"].pop("scales", None)
        cfg["train_dataloader"]["collate_fn"].update(base_size=640, base_size_repeat=None)
        cfg["epochs"] = 20
        cfg["DFINECriterion"] = {"matcher": copy.deepcopy(pilot["RTDETRCriterionv2"]["matcher"])}
        cfg["DFINETransformer"] = copy.deepcopy(pilot["RTDETRTransformerv2"])
        self.assertTrue(all(prep.protocol(pilot, cfg).values()))
        cfg["optimizer"]["lr"] *= 2
        with self.assertRaisesRegex(ValueError, "mismatch"):
            prep.protocol(pilot, cfg)


if __name__ == "__main__":
    unittest.main()
