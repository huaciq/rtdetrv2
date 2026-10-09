"""Bundle existing P1 cases plus Storage Tank center-oracle examples. No inference."""
import argparse
from collections import defaultdict, deque
import csv
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import zipfile

from PIL import Image, ImageDraw

GROUPS = ("center_recovers75", "center_insufficient75")
BINS = ("<8", "8-16", "16-32", ">=32")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def dump(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows, fields):
    with Path(path).open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def category_key(name):
    return "".join(c for c in name.casefold() if c.isalnum())


def box(value):
    values = [float(x) for x in value]
    if len(values) != 4 or not all(math.isfinite(v) for v in values) or min(values[2:]) <= 0:
        raise ValueError(f"Invalid bbox: {value}")
    return values


def iou(a, b):
    a, b = box(a), box(b)
    area = max(0., min(a[0]+a[2], b[0]+b[2])-max(a[0], b[0])) * max(0., min(a[1]+a[3], b[1]+b[3])-max(a[1], b[1]))
    return area/(a[2]*a[3]+b[2]*b[3]-area)


def centered_box(pred, gt):
    p, g = box(pred), box(gt)
    return [g[0]+g[2]/2-p[2]/2, g[1]+g[3]/2-p[3]/2, p[2], p[3]]


def center_group(before, after):
    if not (.5 <= before < .75):
        return None
    return GROUPS[0] if after >= .75 else GROUPS[1]


def verify_validation_gt(truth, gt_path, provenance, previous, p1_annotations=None):
    """Compare GT content only after linking files to their own recorded byte SHA.

    P1 hashes the configured original val.json; recovery can hash a re-serialized
    export validation_gt.json. Those SHA values need not be equal. A mismatching
    pair must be resolved to actual source files, never silently ignored.
    """
    analysis_sha = previous["annotations_sha256"]
    p1_sha = provenance.get("validation_sha256")
    original = Path(previous["annotations_source"])
    result = {"analysis_recorded_sha256": analysis_sha, "P1_recorded_sha256": p1_sha,
              "analysis_copy_sha256": sha(gt_path), "same_recorded_byte_sha": p1_sha == analysis_sha,
              "content_comparison": "exact parsed JSON; object key order/formatting ignored; arrays/values preserved"}
    def validate(path, expected, label):
        actual = sha(path)
        if actual != expected:
            raise ValueError(f"{label} GT source byte hash changed: {path}; expected {expected}, actual {actual}")
        if read_json(path) != truth:
            raise ValueError(f"{label} validation GT content differs from analysis GT: {path}")
        result[label+"_source"] = str(path)
        result[label+"_source_verified"] = True
    # The saved analysis copy can itself be a byte-identical source copy.
    if original.is_file():
        validate(original, analysis_sha, "Original")
    elif result["analysis_copy_sha256"] == analysis_sha:
        validate(gt_path, analysis_sha, "Original")
    else:
        raise FileNotFoundError(f"Cannot verify recorded error-analysis GT source: {original}; locate its byte-identical copy")
    if not p1_sha:
        if p1_annotations:
            raise ValueError("P1 validation byte hash missing; cannot verify --p1-annotations")
        result["P1_source_verified"] = False
        return result
    if p1_annotations:
        validate(Path(p1_annotations).resolve(), p1_sha, "P1")
    elif p1_sha == analysis_sha:
        result["P1_source"] = result["Original_source"]
        result["P1_source_verified"] = True
    elif result["analysis_copy_sha256"] == p1_sha:
        validate(gt_path, p1_sha, "P1")
    else:
        repo = Path(__file__).resolve().parents[1]
        config = Path(provenance.get("config", ""))
        config = config if config.is_absolute() else repo/config
        if not config.is_file():
            raise FileNotFoundError("Recorded GT hashes differ and P1 config is unavailable; provide --p1-annotations with the original P1 val.json")
        expected_config_sha = provenance.get("config_file_sha256")
        if expected_config_sha and sha(config) != expected_config_sha:
            raise ValueError("P1 config bytes changed; provide --p1-annotations with its original val.json")
        # Reuse the exact P1 include/override resolver; no model is built.
        from localization_diagnosis import resolved_config
        p1_path = Path(resolved_config(config)["val_dataloader"]["dataset"]["ann_file"])
        p1_path = p1_path if p1_path.is_absolute() else repo/p1_path
        if not p1_path.is_file():
            raise FileNotFoundError(f"P1 original validation GT missing: {p1_path}; provide --p1-annotations with its byte-identical relocated copy")
        validate(p1_path, p1_sha, "P1")
    result["P1_and_analysis_content_equal"] = True
    return result


def select_focus(errors, indexed_ids, count):
    """Missing-index first, round-robin input-size bins; distinct images per group.

    Within each bin use image_id/gt_id, not largest oracle gain. Illustrative,
    deterministic selection; never a statistical sample or parameter search.
    """
    selections, candidates = {}, {}
    used_images = set()
    for group in GROUPS:
        pool = [r for r in errors if r["center_group"] == group and category_key(r["category_name"]) == "storagetank"]
        candidates[group] = {"objects": len(pool), "images": len({r["image_id"] for r in pool})}
        selected = []
        selected_images = set()
        # Prefer disjoint images, but do not lose a group just because different
        # tanks in the same SAR image have different error mechanisms.
        for existing, shared in ((False, False), (False, True), (True, False), (True, True)):
            buckets = defaultdict(deque)
            for row in sorted(pool, key=lambda r: (r["image_id"], r["gt_id"])):
                if (row["gt_id"] in indexed_ids) == existing and (row["image_id"] in used_images) == shared:
                    buckets[row.get("size_group", "unknown")].append(row)
            keys = list(BINS)+sorted(set(buckets)-set(BINS))
            while len(selected) < count and any(buckets[k] for k in keys):
                for key in keys:
                    while buckets[key] and buckets[key][0]["image_id"] in selected_images:
                        buckets[key].popleft()
                    if buckets[key] and len(selected) < count:
                        row = buckets[key].popleft()
                        selected.append(row)
                        selected_images.add(row["image_id"])
                        used_images.add(row["image_id"])
        selections[group] = selected
    return selections, candidates


def render_case(image, gt, pred, corrected, caption, folder):
    # Left view is actual prediction, right view is hypothetical GT-center oracle.
    def draw_view(oracle):
        canvas = image.convert("RGB")
        draw = ImageDraw.Draw(canvas)
        for bounds, color, width in ((gt["bbox"], "lime", 2),):
            x, y, w, h = bounds
            draw.rectangle((x, y, x+w, y+h), outline=color, width=width)
        if pred is not None:
            bounds = corrected if oracle else pred["bbox"]
            x, y, w, h = bounds
            draw.rectangle((x, y, x+w, y+h), outline="cyan" if oracle else "red", width=2)
        return canvas
    actual, oracle = draw_view(False), draw_view(True)
    actual.save(folder/"actual_boxes.png")
    x, y, w, h = gt["bbox"]
    pad = max(w, h, 24.)
    bounds = (max(0, math.floor(x-pad)), max(0, math.floor(y-pad)),
              min(image.width, math.ceil(x+w+pad)), min(image.height, math.ceil(y+h+pad)))
    if bounds[2] <= bounds[0] or bounds[3] <= bounds[1]:
        raise ValueError(f"GT does not intersect original image: {gt['id']}")
    def crop(canvas):
        result = canvas.crop(bounds)
        factor = 640/max(result.size)
        return result.resize((max(1, round(result.width*factor)), max(1, round(result.height*factor))), Image.Resampling.NEAREST)
    left, right = crop(actual), crop(oracle)
    pane_width = max(left.width, 400)
    panel = Image.new("RGB", (pane_width*2, left.height+72), "black")
    panel.paste(left, (0, 72)); panel.paste(right, (pane_width, 72))
    draw = ImageDraw.Draw(panel)
    draw.text((5, 4), caption, fill="white")
    draw.text((5, 49), "Actual: GT=green, prediction=red", fill="white")
    draw.text((pane_width+5, 49), "Oracle only: GT=green, centered=cyan", fill="white")
    panel.save(folder/"center_comparison.png")
    return list(bounds)


def package(args):
    diag, analysis, image_root = Path(args.diagnosis).resolve(), Path(args.analysis).resolve(), Path(args.images).resolve()
    out = Path(args.output).resolve()
    archive = Path(str(out)+".zip")
    if archive.exists() or archive.with_suffix(".zip.partial").exists():
        raise FileExistsError(f"Preserve existing archive: {archive}")
    if out.exists() and any(p.name not in ("console.log", "launcher.pid") for p in out.iterdir()):
        raise FileExistsError(f"Use a fresh package directory: {out}")
    if args.per_group <= 0:
        raise ValueError("per-group must be positive")
    index_path, errors_path = diag/"example_index.csv", diag/"matched_box_errors.csv"
    gt_path, pred_path = analysis/"validation_gt.json", analysis/"predictions.json"
    index, table, truth, predictions = read_csv(index_path), read_csv(errors_path), read_json(gt_path), read_json(pred_path)
    provenance_path = diag/"provenance.json"
    provenance = read_json(provenance_path) if provenance_path.exists() else {}
    prediction_sha = sha(pred_path)
    if provenance.get("prediction_sha256") and provenance["prediction_sha256"] != prediction_sha:
        raise ValueError("P1 provenance prediction hash differs from provided predictions")
    analysis_manifest_path = analysis/"analysis_manifest.json"
    gt_verification = {"status": "no_analysis_manifest"}
    if analysis_manifest_path.exists():
        previous = read_json(analysis_manifest_path)
        if previous["predictions_sha256"] != prediction_sha:
            raise ValueError("Error-analysis manifest prediction hash mismatch")
        gt_verification = verify_validation_gt(truth, gt_path, provenance, previous, getattr(args, "p1_annotations", None))
    images = {r["id"]: r for r in truth["images"]}
    gts = {r["id"]: r for r in truth["annotations"]}
    cats = {r["id"]: r["name"] for r in truth["categories"]}
    if len(images) != len(truth["images"]) or len(gts) != len(truth["annotations"]):
        raise ValueError("Duplicate original image/GT IDs")
    def checked_prediction(pid, gt):
        if not 0 <= pid < len(predictions):
            raise ValueError(f"Invalid original prediction_id {pid}")
        p = predictions[pid]
        if p["image_id"] != gt["image_id"] or p["category_id"] != gt["category_id"]:
            raise ValueError(f"Wrong prediction link for GT {gt['id']}")
        box(p["bbox"])
        if not math.isfinite(p["score"]):
            raise ValueError("Non-finite score")
        return p
    errors = {}
    for r in table:
        gid, iid, cid, pid = (int(r[k]) for k in ("gt_id", "image_id", "category_id", "prediction_id"))
        gt = gts[gid]
        if gid in errors or gt["image_id"] != iid or gt["category_id"] != cid or category_key(cats[cid]) != category_key(r["category_name"]):
            raise ValueError(f"Invalid/duplicate error-table link: GT {gid}")
        p = checked_prediction(pid, gt)
        if box(json.loads(r["bbox"])) != box(gt["bbox"]):
            raise ValueError(f"GT bbox mismatch: {gid}")
        before, after = iou(p["bbox"], gt["bbox"]), iou(centered_box(p["bbox"], gt["bbox"]), gt["bbox"])
        for key, actual in (("iou", before), ("iou_fix_center", after), ("gain_fix_center", after-before), ("score", p["score"])):
            if not math.isclose(float(r[key]), actual, abs_tol=1e-6, rel_tol=0):
                raise ValueError(f"CSV/source mismatch {key}: GT {gid}")
        errors[gid] = dict(r, gt_id=gid, image_id=iid, category_id=cid, prediction_id=pid,
                          iou=before, iou_fix_center=after, gain_fix_center=after-before, center_group=center_group(before, after))
    indexed = defaultdict(list)
    for r in index:
        gid, iid = int(r["gt_id"]), int(r["image_id"])
        gt = gts[gid]
        if gt["image_id"] != iid or category_key(cats[gt["category_id"]]) != category_key(r["category_name"]):
            raise ValueError(f"Wrong existing example index: GT {gid}")
        indexed[gid].append(r)
    selections, candidates = select_focus(list(errors.values()), set(indexed), args.per_group)
    print("Validated source links; Storage Tank selected: "+json.dumps({g: len(selections[g]) for g in GROUPS}), flush=True)
    memberships = {r["gt_id"]: group for group in GROUPS for r in selections[group]}
    gids = sorted(set(indexed) | set(memberships))
    if not gids:
        raise ValueError("No existing or eligible supplemental cases")
    by_image = defaultdict(list)
    for pid, p in enumerate(predictions):
        by_image[p["image_id"]].append((pid, p))
    # FN75 is not necessarily TP50. Preserve its original visualized coverage box.
    old_gt_path = analysis/"gt_localization.csv"
    old_gt_rows = {int(r["gt_id"]): r for r in read_csv(old_gt_path)} if old_gt_path.exists() else {}
    records, original_paths = [], {}
    for gid in gids:
        gt, row = gts[gid], errors.get(gid)
        image = images[gt["image_id"]]
        path = (image_root/image["file_name"]).resolve()
        if not path.is_relative_to(image_root) or not path.is_file():
            raise FileNotFoundError(f"Original image missing/outside root: {path}")
        with Image.open(path) as im:
            if im.size != (image["width"], image["height"]):
                raise ValueError(f"GT/image dimensions differ: image {image['id']}")
        original_paths[image["id"]] = path
        if row is not None:
            pid, kind = row["prediction_id"], "COCO_one_to_one_TP50"
        elif old_gt_rows.get(gid, {}).get("best_same_prediction_id", "") != "":
            pid, kind = int(old_gt_rows[gid]["best_same_prediction_id"]), "existing_FN75_same_class_coverage_not_TP"
        else:
            pool = [(pid, p) for pid, p in by_image[image["id"]] if p["category_id"] == gt["category_id"]]
            pid = max(pool, key=lambda item: (iou(item[1]["bbox"], gt["bbox"]), item[1]["score"], -item[0]))[0] if pool else None
            kind = "reconstructed_same_class_coverage_not_TP" if pid is not None else "no_same_class_prediction"
        p = checked_prediction(pid, gt) if pid is not None else None
        centered = centered_box(p["bbox"], gt["bbox"]) if p is not None else None
        records.append({"gt_id": gid, "image_id": image["id"], "category_id": gt["category_id"],
            "category_name": cats[gt["category_id"]], "original_file_name": image["file_name"], "GT": gt,
            "original_prediction_id": pid, "prediction": p, "prediction_basis": kind,
            "center_oracle_bbox_xywh": centered, "iou": iou(p["bbox"], gt["bbox"]) if p is not None else None,
            "iou_fix_center": iou(centered, gt["bbox"]) if p is not None else None,
            "matched_box_error_row": row, "focus_group": memberships.get(gid),
            "existing_index_entries": indexed.get(gid, []), "manual_review": "pending"})
    out.mkdir(parents=True, exist_ok=True)
    (out/"images").mkdir(); (out/"cases").mkdir(); (out/"source").mkdir()
    image_files = {}
    for iid, source in sorted(original_paths.items()):
        relative = f"images/image_{iid}{source.suffix}"
        shutil.copy2(source, out/relative)
        image_files[iid] = {"file": relative, "original_file_name": images[iid]["file_name"], "sha256": sha(source)}
        if sha(out/relative) != image_files[iid]["sha256"]:
            raise RuntimeError("Original image copy byte mismatch")
    entries = []
    for case_number, record in enumerate(records, 1):
        gid, iid = record["gt_id"], record["image_id"]
        folder = out/"cases"/f"GT_{gid}"
        folder.mkdir()
        record["image"] = image_files[iid]
        with Image.open(original_paths[iid]) as im:
            caption = f"image_id={iid} GT={gid} class={record['category_name']}\nIoU={record['iou']} -> centered={record['iou_fix_center']}"
            record["crop_xyxy_original_pixels"] = render_case(im, record["GT"], record["prediction"], record["center_oracle_bbox_xywh"], caption, folder)
        preserved = []
        for n, source_entry in enumerate(record["existing_index_entries"]):
            path = Path(source_entry["image_path"])
            path = path if path.is_absolute() else diag/path
            copies = []
            for suffix, original in (("view", path), ("crop", path.with_name(path.stem+"_crop"+path.suffix))):
                if original.is_file():
                    with Image.open(original) as view:
                        view.verify()
                    target = folder/f"existing_{n}_{suffix}{original.suffix}"
                    shutil.copy2(original, target)
                    copies.append(target.relative_to(out).as_posix())
            preserved.append({"sample_id": source_entry["sample_id"], "copied_files": copies,
                              "original_visualization_missing": not path.is_file()})
        record["preserved_visualizations"] = preserved
        dump(folder/"record.json", record)
        entries.append({"gt_id": gid, "image_id": iid, "category_name": record["category_name"],
            "original_file_name": record["original_file_name"], "focus_group": record["focus_group"] or "existing_index",
            "new_to_example_index": gid not in indexed, "prediction_id": record["original_prediction_id"],
            "prediction_basis": record["prediction_basis"], "iou": record["iou"], "iou_fix_center": record["iou_fix_center"],
            "gain_fix_center": record["iou_fix_center"]-record["iou"] if record["iou"] is not None else None,
            "size_group": record["matched_box_error_row"].get("size_group", "") if record["matched_box_error_row"] else "",
            "original_image": image_files[iid]["file"], "annotated_image": f"cases/GT_{gid}/actual_boxes.png",
            "comparison_image": f"cases/GT_{gid}/center_comparison.png", "record": f"cases/GT_{gid}/record.json"})
        if case_number % 10 == 0 or case_number == len(records):
            print(f"Packaged cases {case_number}/{len(records)}", flush=True)
    write_csv(out/"case_index.csv", entries, list(entries[0]))
    write_csv(out/"storage_tank_center_cases.csv", [r for r in entries if r["focus_group"] in GROUPS], list(entries[0]))
    selected_images = set(original_paths)
    subset_gt = dict(truth, images=[r for r in truth["images"] if r["id"] in selected_images],
                     annotations=[r for r in truth["annotations"] if r["image_id"] in selected_images])
    dump(out/"validation_gt.json", subset_gt)  # all GT in selected images, original IDs/filenames unchanged
    subset = [(pid, p) for pid, p in enumerate(predictions) if p["image_id"] in selected_images]
    dump(out/"predictions.json", [p for _, p in subset])  # all original scores/categories in selected images
    write_csv(out/"prediction_index.csv", [{"packaged_row_index": n, "original_prediction_id": pid, "image_id": p["image_id"]}
                                          for n, (pid, p) in enumerate(subset)], ["packaged_row_index", "original_prediction_id", "image_id"])
    for source in (index_path, errors_path, provenance_path, diag/"manual_review.csv", old_gt_path, analysis_manifest_path):
        if source.exists():
            shutil.copy2(source, out/"source"/source.name)
    manifest = {"status": "complete", "inference_or_training": False, "per_group_requested": args.per_group,
        "selection_rule": "Storage Tank, original IoU [0.5,0.75); effective iff centered IoU >=0.75; ineffective means insufficient to restore 0.75, NOT zero gain",
        "sampling": "missing example_index first; input-size round-robin, deterministic image_id/gt_id order; unique images within group; prefer disjoint across groups",
        "shared_focus_image_ids": sorted({r["image_id"] for r in selections[GROUPS[0]]} & {r["image_id"] for r in selections[GROUPS[1]]}),
        "candidate_counts": candidates, "focus_selected": {g: len(selections[g]) for g in GROUPS},
        "focus_new_to_index": {g: sum(r["gt_id"] not in indexed for r in selections[g]) for g in GROUPS},
        "cases": len(records), "original_images": len(original_paths), "existing_index_rows": len(index),
        "input_sha256": {"example_index": sha(index_path), "matched_box_errors": sha(errors_path),
                         "full_predictions": prediction_sha, "full_validation_gt": sha(gt_path)},
        "P1_provenance_available": bool(provenance), "manual_review": "pending",
        "validation_gt_verification": gt_verification,
        "diagnosis_source": str(diag), "analysis_source": str(analysis), "images_source": str(image_root),
        "packager_script_sha256": sha(__file__),
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[1], text=True).strip()}
    dump(out/"manifest.json", manifest)
    (out/"README.md").write_text(
        "# Localization case bundle\n\n原图保存在 images/，逐图原始 image_id 与原文件名见 case_index.csv；文件字节未改。\n"
        "每个 cases/GT_ID/record.json 保存完整GT、真实预测、原始全预测数组 prediction_id、CSV误差和中心oracle。\n"
        "actual_boxes.png：绿色GT、红色真实预测。center_comparison.png：左真实框，右青色GT中心oracle（不是模型输出）。\n"
        "storage_tank_center_cases.csv：每组目标约10张，原IoU∈[.5,.75)，修正中心后≥.75为center_recovers75，仍<.75为center_insufficient75；后者不等于零增益。\n"
        "先补未在example_index列出的GT，再按输入短边轮转；不按最大增益挑选。组内不重复image_id，优先组间也不重复；必要时可用同图不同GT，共享图ID见manifest。不足不复制补足。\n"
        "validation_gt.json/predictions.json保存所选图像的全部GT/预测，保留原ID与bbox、score；COCO GT file_name不改，便携原图路径见case_index.csv/image映射。\n"
        "prediction_index.csv映射子集行号到原始prediction_id。FN75的覆盖框不能称为TP50；预测依据见record.json。\n"
        "source/保留原索引/误差/复核资料；这些原CSV可能含历史绝对路径，但新case_index.csv路径全部相对压缩包根目录。\n"
        "这是示例包，不能据此估计错误比例或断定模糊边界、散射偏移、标签质量；原人工审阅资料保留，不自动改为已复核。\n",
        encoding="utf-8")
    inventory = [{"file": p.relative_to(out).as_posix(), "bytes": p.stat().st_size, "sha256": sha(p)}
                 for p in sorted(out.rglob("*")) if p.is_file() and p.name not in ("console.log", "launcher.pid")]
    dump(out/"file_manifest.json", inventory)
    partial = archive.with_suffix(".zip.partial")
    with zipfile.ZipFile(partial, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for item in inventory+[{"file": "file_manifest.json"}]:
            z.write(out/item["file"], arcname=out.name+"/"+item["file"])
    with zipfile.ZipFile(partial) as z:
        if z.testzip() is not None:
            raise RuntimeError("ZIP CRC validation failed")
    partial.rename(archive)
    dump(out/"COMPLETE.json", {"status": "complete", "archive": str(archive), "archive_sha256": sha(archive),
                              "focus_selected": manifest["focus_selected"], "cases": len(records), "manual_review": "pending"})
    print(json.dumps({"archive": str(archive), "focus_selected": manifest["focus_selected"], "cases": len(records)}, ensure_ascii=False), flush=True)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnosis", required=True, help="P1 output containing example_index/matched_box_errors.csv")
    parser.add_argument("--analysis", required=True, help="Original error-analysis output with full predictions/validation_gt.json")
    parser.add_argument("--images", required=True, help="Original validation image root")
    parser.add_argument("--output", required=True, help="Fresh directory; ZIP is written beside it")
    parser.add_argument("--per-group", type=int, default=10)
    parser.add_argument("--p1-annotations", help="Optional relocated original P1 val.json; must match P1's recorded byte SHA")
    package(parser.parse_args())


if __name__ == "__main__":
    main()
