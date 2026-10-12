"""Synthetic protocol/label/scene/review tests; never builds a detector."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image

import error_structure_analysis as structure


def fixture(root):
    image_root, source = root/"images", root/"source"
    image_root.mkdir(); source.mkdir()
    cats = [{"id": 0, "name": "Bridge"}, {"id": 1, "name": "Harbor"}, {"id": 2, "name": "Storage Tank"}]
    gt = {"info": {}, "images": [], "annotations": [], "categories": cats}
    preds = []
    for iid in range(1, 8):
        Image.new("RGB", (200, 200), (30, 30, iid)).save(image_root/f"{iid}.png")
        gt["images"].append({"id": iid, "file_name": f"{iid}.png", "width": 200, "height": 200})
        gt["annotations"].append({"id": iid, "image_id": iid, "category_id": 0, "bbox": [20, 20, 20, 20], "area": 400, "iscrowd": 0})
    def pred(iid, box, score, cat=0):
        preds.append({"image_id": iid, "category_id": cat, "bbox": box, "score": score})
    pred(1, [20, 20, 20, 20], .95)   # TP
    pred(1, [20, 20, 20, 20], .8)    # Dupe
    pred(2, [150, 150, 10, 10], .99) # Bkg, susceptible to missing annotation
    pred(3, [28, 20, 20, 20], .9)    # Loc (~.43)
    pred(4, [20, 20, 20, 20], .9, 1) # Cls
    pred(5, [28, 20, 20, 20], .9, 1) # Both
    pred(6, [20, 20, 20, 20], .2)    # low score TP: FN only at high-score scope
    # GT7 is crowd and should not create an FN; matching pred is ignored.
    gt["annotations"][-1]["iscrowd"] = 1
    pred(7, [20, 20, 20, 20], .9)
    structure.base.dump(source/"validation_gt.json", gt)
    structure.base.dump(source/"predictions.json", preds)
    structure.base.dump(source/"analysis_manifest.json", {"predictions_sha256": structure.base.sha(source/"predictions.json"),
        "annotations_source": str(source/"validation_gt.json"), "annotations_sha256": structure.base.sha(source/"validation_gt.json"),
        "git_commit": "synthetic", "images_root": str(image_root)})
    return gt, preds, image_root, source


class StructureTests(unittest.TestCase):
    def test_offline_full_report_counts_confidence_crowd_and_case_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            gt, preds, images, source = fixture(root)
            out = root/"out"
            argv = ["error_structure_analysis.py", "--analysis", str(source), "--images", str(images), "--output", str(out)]
            with patch("sys.argv", argv):
                structure.main()
            rows = structure.csv_rows(out/"class_statistics.csv")
            high = next(r for r in rows if r["category_id"] == "0" and r["score_scope"] == "score_ge_0_5" and r["iou_threshold"] == "0.5")
            self.assertEqual([int(high[k]) for k in ("GT", "TP", "FP", "FN", "Bkg", "Loc", "Dupe")], [6, 1, 3, 5, 1, 1, 1])
            low = next(r for r in rows if r["category_id"] == "0" and r["score_scope"] == "all_scores" and r["iou_threshold"] == "0.5")
            self.assertEqual([int(low[k]) for k in ("TP", "FN")], [2, 4])
            harbor = next(r for r in rows if r["category_id"] == "1" and r["score_scope"] == "score_ge_0_5" and r["iou_threshold"] == "0.5")
            self.assertEqual([int(harbor[k]) for k in ("Cls", "Both", "FP", "GT")], [1, 1, 2, 0])
            complete = structure.load(out/"COMPLETE.json")
            self.assertEqual(complete["status"], "complete")
            self.assertEqual(complete["scene_coverage"]["scene_type"], 0)
            self.assertTrue((out/"ERROR_STRUCTURE_REPORT.md").is_file())
            self.assertEqual(len(structure.csv_rows(out/"scene_statistics.csv")), 0)
            vis = structure.csv_rows(out/"visualization_index.csv")
            bkg = next(r for r in vis if r["kind"] == "Bkg")
            self.assertEqual(bkg["prediction_id"], "2")
            self.assertTrue((out/bkg["file"]).exists())
            self.assertIn("不能直接证明SAR", (out/"ERROR_STRUCTURE_REPORT.md").read_text(encoding="utf-8"))
            # Inputs remain byte-identical; all output is separate.
            self.assertEqual(structure.load(source/"predictions.json"), preds)
            self.assertEqual(structure.load(source/"validation_gt.json"), gt)
            with patch("sys.argv", argv), self.assertRaises(FileExistsError):
                structure.main()

    def test_review_suspicion_is_separate_from_annotated_fp(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            gt, preds, images, source = fixture(root)
            out = root/"eval"; out.mkdir()
            _, matches, _ = structure.base.coco_evaluate(source/"validation_gt.json", preds, out)
            sha = structure.base.sha(source/"predictions.json")
            review = root/"reviews.csv"
            row = {"prediction_id": 2, "image_id": 2, "prediction_sha256": sha, "review_status": "suspected_missing_annotation",
                   "reviewer": "synthetic-reviewer", "evidence": "synthetic test only, not a SAR review"}
            structure.base.write_csv(review, [row])
            reviews = structure.review_map(review, preds, sha)
            ds, gs = structure.records(gt, preds, matches, structure.scene_map(gt), reviews)
            high = [d for d in ds if d["iou_threshold"] == .5 and d["score"] >= .5 and d["category_id"] == 0]
            truths = [dict(g, miss=g["miss_score_ge_0_5"]) for g in gs if g["iou_threshold"] == .5 and g["category_id"] == 0]
            c = structure.counts(high, truths)
            self.assertEqual(c["FP"], 3)
            self.assertEqual(c["suspected_missing_annotation"], 1)
            self.assertEqual(c["review_confirmed_background"], 0)
            row["prediction_sha256"] = "wrong-export"
            structure.base.write_csv(review, [row])
            with self.assertRaisesRegex(ValueError, "different prediction export"):
                structure.review_map(review, preds, sha)
            row["prediction_sha256"] = sha; row["evidence"] = ""
            structure.base.write_csv(review, [row])
            with self.assertRaisesRegex(ValueError, "require reviewer and evidence"):
                structure.review_map(review, preds, sha)

    def test_partial_explicit_scene_mapping_no_filename_inference(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            gt, _, _, _ = fixture(root)
            path = root/"scenes.csv"
            structure.base.write_csv(path, [{"image_id": 1, "scene_id": "site_A", "scene_type": "explicit_test_tag"}])
            mapped = structure.scene_map(gt, path)
            self.assertEqual(mapped[1]["scene_type"], "explicit_test_tag")
            self.assertEqual(mapped[2]["scene_type"], "")
            gt["images"][0]["scene_type"] = None
            self.assertEqual(structure.scene_map(gt)[1]["scene_type"], "")

    def test_classification_priority_and_no_unproved_duplicate(self):
        self.assertEqual(structure.fp_kind(.2, .9, False, .5), "Cls")
        self.assertEqual(structure.fp_kind(.8, .8, True, .5), "Dupe")
        self.assertEqual(structure.fp_kind(.8, .8, False, .5), "Unresolved")
        self.assertEqual(structure.fp_kind(.15, .3, False, .5), "Loc")
        self.assertEqual(structure.fp_kind(0., .3, False, .5), "Both")
        self.assertEqual(structure.fp_kind(0., .09, False, .5), "Bkg")

    def test_empty_predictions_all_non_crowd_gt_missed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            gt, _, _, source = fixture(root)
            out = root/"eval"; out.mkdir()
            _, matches, _ = structure.base.coco_evaluate(source/"validation_gt.json", [], out)
            ds, gs = structure.records(gt, [], matches, structure.scene_map(gt), {})
            self.assertEqual(ds, [])
            c = structure.counts([], [dict(g, miss=g["miss_all_scores"]) for g in gs if g["iou_threshold"] == .5])
            self.assertEqual([c[k] for k in ("TP", "FP", "FN")], [0, 0, 6])

    def test_input_hash_change_and_nonbaseline_export_rejected_before_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, preds, images, source = fixture(root)
            out = root/"out"
            argv = ["error_structure_analysis.py", "--analysis", str(source), "--images", str(images), "--output", str(out)]
            changed = [dict(p) for p in preds]
            changed[0]["score"] = .1
            structure.base.dump(source/"predictions.json", changed)
            with patch("sys.argv", argv), self.assertRaisesRegex(ValueError, "do not match analysis manifest"):
                structure.main()
            self.assertFalse(out.exists())
            structure.base.dump(source/"predictions.json", preds)
            structure.base.dump(source/"export_metadata.json", {"strict_load": True,
                "resolved_config": {"RTDETRTransformerv2": {"sber": True}}})
            with patch("sys.argv", argv+["--export-dir", str(source)]), self.assertRaisesRegex(ValueError, "E0 baseline export"):
                structure.main()
            self.assertFalse(out.exists())


if __name__ == "__main__":
    unittest.main()
