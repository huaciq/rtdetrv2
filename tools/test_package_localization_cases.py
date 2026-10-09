"""Synthetic checks for portable case packaging; no SAR performance claims."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
import zipfile

from PIL import Image

import package_localization_cases as bundle


def fixture(root):
    diagnosis, analysis, images = root/"diagnosis", root/"analysis", root/"originals"
    for folder in (diagnosis, analysis, images):
        folder.mkdir()
    truth = {"info": {}, "images": [], "annotations": [], "categories": [
        {"id": 5, "name": "storage_tank"}, {"id": 7, "name": "harbor"}]}
    predictions, errors, index = [], [], []
    for n in range(24):
        iid, gid, w = 100+n, 1000+n, (6, 10, 20, 40)[n % 4]
        filename = f"scene_{iid}.jpg" if n == 0 else f"scene_{iid}.png"
        Image.new("RGB", (200, 160), (n, 40, 100)).save(images/filename)
        truth["images"].append({"id": iid, "file_name": filename, "width": 200, "height": 160})
        gt = {"id": gid, "image_id": iid, "category_id": 5, "bbox": [40, 40, w, w], "area": w*w, "iscrowd": 0}
        truth["annotations"].append(gt)
        bbox = [40+.2*w, 40, w, w] if n < 12 else [40+.05*w, 40+.05*w, 1.2*w, 1.2*w]
        pred = {"image_id": iid, "category_id": 5, "bbox": bbox, "score": .9}
        predictions.append(pred)
        before = bundle.iou(bbox, gt["bbox"])
        after = bundle.iou(bundle.centered_box(bbox, gt["bbox"]), gt["bbox"])
        errors.append({"gt_id": gid, "image_id": iid, "category_id": 5, "category_name": "Storage Tank",
            "bbox": json.dumps(gt["bbox"]), "prediction_id": n, "score": .9, "iou": before,
            "iou_fix_center": after, "gain_fix_center": after-before, "size_group": bundle.BINS[n % 4]})
        if n in (0, 1, 12, 13):
            view = diagnosis/f"old_{gid}.png"
            Image.new("RGB", (200, 160), "gray").save(view)
            index.append({"sample_id": f"old_{gid}", "gt_id": gid, "image_id": iid,
                          "category_name": "Storage Tank", "image_path": str(view), "source": "existing"})
    # Existing FN points outside P1 directory; preserve it and its original coverage prediction.
    Image.new("RGB", (200, 160), "navy").save(images/"harbor.png")
    truth["images"].append({"id": 200, "file_name": "harbor.png", "width": 200, "height": 160})
    truth["annotations"].append({"id": 9000, "image_id": 200, "category_id": 7, "bbox": [30, 30, 10, 10], "area": 100, "iscrowd": 0})
    predictions.append({"image_id": 200, "category_id": 7, "bbox": [60, 60, 10, 10], "score": .6})
    old = analysis/"FN75.png"
    Image.new("RGB", (200, 160), "gray").save(old)
    Image.new("RGB", (100, 100), "gray").save(analysis/"FN75_crop.png")
    index.append({"sample_id": "old_FN75_9000", "gt_id": 9000, "image_id": 200,
                  "category_name": "Harbor", "image_path": str(old), "source": "existing FN75"})
    bundle.dump(analysis/"validation_gt.json", truth)
    bundle.dump(root/"original_validation.json", truth)
    bundle.dump(analysis/"predictions.json", predictions)
    bundle.write_csv(analysis/"gt_localization.csv", [{"gt_id": 9000, "best_same_prediction_id": 24}], ["gt_id", "best_same_prediction_id"])
    bundle.write_csv(diagnosis/"matched_box_errors.csv", errors, list(errors[0]))
    bundle.write_csv(diagnosis/"example_index.csv", index, list(index[0]))
    bundle.dump(diagnosis/"provenance.json", {"prediction_sha256": bundle.sha(analysis/"predictions.json"),
        "validation_sha256": bundle.sha(root/"original_validation.json")})
    bundle.dump(analysis/"analysis_manifest.json", {"predictions_sha256": bundle.sha(analysis/"predictions.json"),
        "annotations_source": str(root/"original_validation.json"), "annotations_sha256": bundle.sha(root/"original_validation.json")})
    return SimpleNamespace(diagnosis=str(diagnosis), analysis=str(analysis), images=str(images),
                           output=str(root/"bundle"), per_group=10), errors


class PackageTests(unittest.TestCase):
    def test_portable_full_bundle_and_twenty_missing_focus_cases(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args, errors = fixture(root)
            manifest = bundle.package(args)
            self.assertEqual(manifest["focus_selected"], dict.fromkeys(bundle.GROUPS, 10))
            self.assertEqual(manifest["focus_new_to_index"], dict.fromkeys(bundle.GROUPS, 10))
            self.assertEqual(manifest["cases"], 25)  # indexed4 + supplemental20 + FN1
            out = Path(args.output)
            rows = bundle.read_csv(out/"storage_tank_center_cases.csv")
            self.assertEqual(len(rows), 20)
            for row in rows:
                self.assertFalse(Path(row["original_image"]).is_absolute())
                self.assertTrue((out/row["original_image"]).is_file())
                self.assertTrue((out/row["annotated_image"]).is_file())
                self.assertEqual(int(row["image_id"]), int(row["gt_id"])-900)
                if row["focus_group"] == bundle.GROUPS[0]:
                    self.assertGreaterEqual(float(row["iou_fix_center"]), .75)
                else:
                    self.assertLess(float(row["iou_fix_center"]), .75)
                    self.assertGreater(float(row["gain_fix_center"]), 0)  # insufficient != zero gain
            self.assertEqual(bundle.sha(root/"originals"/"scene_100.jpg"), bundle.sha(out/"images"/"image_100.jpg"))
            fn = bundle.read_json(out/"cases"/"GT_9000"/"record.json")
            self.assertEqual(fn["image_id"], 200)
            self.assertEqual(fn["original_prediction_id"], 24)
            self.assertIn("not_TP", fn["prediction_basis"])
            self.assertEqual(fn["manual_review"], "pending")
            self.assertEqual(len(fn["preserved_visualizations"][0]["copied_files"]), 2)
            self.assertEqual(bundle.read_json(out/"validation_gt.json")["images"][0]["id"], 100)
            with zipfile.ZipFile(str(out)+".zip") as archive:
                self.assertIsNone(archive.testzip())
                for item in bundle.read_json(out/"file_manifest.json"):
                    import hashlib
                    self.assertEqual(hashlib.sha256(archive.read("bundle/"+item["file"])).hexdigest(), item["sha256"])
            with self.assertRaises(FileExistsError):
                bundle.package(args)

    def test_missing_old_view_is_explicit_and_reconstructed(self):
        with tempfile.TemporaryDirectory() as tmp:
            args, _ = fixture(Path(tmp))
            (Path(args.analysis)/"FN75.png").unlink()
            bundle.package(args)
            record = bundle.read_json(Path(args.output)/"cases"/"GT_9000"/"record.json")
            self.assertTrue(record["preserved_visualizations"][0]["original_visualization_missing"])
            self.assertTrue((Path(args.output)/"cases"/"GT_9000"/"actual_boxes.png").exists())

    def test_wrong_csv_oracle_rejected_before_writing_package(self):
        with tempfile.TemporaryDirectory() as tmp:
            args, errors = fixture(Path(tmp))
            errors[0]["iou_fix_center"] = .5
            bundle.write_csv(Path(args.diagnosis)/"matched_box_errors.csv", errors, list(errors[0]))
            with self.assertRaisesRegex(ValueError, "CSV/source mismatch"):
                bundle.package(args)
            self.assertFalse(Path(args.output).exists())

    def test_wrong_prediction_file_hash_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            args, _ = fixture(Path(tmp))
            path = Path(args.analysis)/"predictions.json"
            predictions = bundle.read_json(path)
            predictions[0]["score"] = .7
            bundle.dump(path, predictions)
            with self.assertRaisesRegex(ValueError, "prediction hash"):
                bundle.package(args)

    def test_wrong_validation_gt_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            args, _ = fixture(Path(tmp))
            path = Path(args.analysis)/"validation_gt.json"
            truth = bundle.read_json(path)
            truth["annotations"][0]["bbox"][0] += 1
            bundle.dump(path, truth)
            with self.assertRaisesRegex(ValueError, "Original validation GT differs"):
                bundle.package(args)

    def test_insufficient_unique_images_and_shared_groups_not_duplicated(self):
        rows = [{"center_group": group, "category_name": "Storage Tank", "image_id": iid,
                 "gt_id": (n+1)*100+iid, "size_group": "<8"}
                for n, group in enumerate(bundle.GROUPS) for iid in (1, 1, 2)]
        selected, candidates = bundle.select_focus(rows, set(), 10)
        self.assertEqual([len(selected[g]) for g in bundle.GROUPS], [2, 2])
        self.assertEqual([candidates[g]["images"] for g in bundle.GROUPS], [2, 2])
        self.assertIsNone(bundle.center_group(.49, .99))
        self.assertEqual(bundle.center_group(.7, .75), bundle.GROUPS[0])
        self.assertEqual(bundle.center_group(.7, .749), bundle.GROUPS[1])


if __name__ == "__main__":
    unittest.main()
