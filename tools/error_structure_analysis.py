"""One read-only, prediction-only error structure analysis. No model imports."""
import argparse
from collections import Counter, defaultdict
import csv
import json
import math
from pathlib import Path
import subprocess

import numpy as np
from PIL import Image, ImageDraw

import baseline_error_analysis as base

KINDS = ("Bkg", "Cls", "Loc", "Dupe", "Both", "Unresolved")
REVIEWS = ("pending", "suspected_missing_annotation", "confirmed_missing_annotation", "confirmed_background", "uncertain")


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def csv_rows(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def scene_map(gt, path=None):
    images = {i["id"]: i for i in gt["images"]}
    rows = csv_rows(path) if path else [dict(image_id=i["id"], scene_id=i.get("scene_id", ""),
        scene_type=i.get("scene_type", "")) for i in images.values()]
    result = {iid: {"scene_id": "", "scene_type": ""} for iid in images}
    seen = set()
    for r in rows:
        iid = int(r["image_id"])
        if iid not in images or iid in seen:
            raise ValueError(f"Unknown/duplicate scene image_id: {iid}")
        seen.add(iid)
        result[iid] = {k: str(r[k]).strip() if r.get(k) is not None else "" for k in ("scene_id", "scene_type")}
    return result


def review_map(path, predictions, prediction_sha):
    result = {}
    for row in csv_rows(path) if path else []:
        pid = int(row["prediction_id"])
        if not 0 <= pid < len(predictions) or pid in result:
            raise ValueError("Unknown/duplicate reviewed original prediction_id")
        if int(row["image_id"]) != predictions[pid]["image_id"] or row["prediction_sha256"] != prediction_sha:
            raise ValueError("Review linked to a different prediction export")
        if row["review_status"] not in REVIEWS:
            raise ValueError(f"Unknown review status: {row['review_status']}")
        if row["review_status"] != "pending" and (not row.get("reviewer", "").strip() or not row.get("evidence", "").strip()):
            raise ValueError("Non-pending reviews require reviewer and evidence")
        result[pid] = row
    return result


def fp_kind(same, any_iou, occupied, threshold):
    if same >= threshold:
        return "Dupe" if occupied else "Unresolved"
    if any_iou >= threshold:
        return "Cls"  # do not hide strong wrong-class overlap behind weak same-class overlap
    if same >= .1:
        return "Loc"
    return "Bkg" if any_iou < .1 else "Both"


def records(gt, predictions, matches, scenes, reviews):
    by_gt, by_pred = defaultdict(list), defaultdict(list)
    for a in gt["annotations"]:
        if not a.get("iscrowd", 0):
            by_gt[a["image_id"]].append(a)
    for pid, p in enumerate(predictions):
        by_pred[p["image_id"]].append((pid, p))
    detections, truths = [], []
    for image in gt["images"]:
        iid = image["id"]
        anns, indexed = by_gt[iid], by_pred[iid]
        ps = [p for _, p in indexed]
        matrix = base.ious(ps, anns)
        for threshold in (.5, .75):
            matched = {int(gid): pid for pid, _ in indexed for gid, ignored in [matches.get((pid, threshold), (0, True))] if gid and not ignored}
            matched_high = {gid for gid, pid in matched.items() if predictions[pid]["score"] >= .5}
            for j, (pid, p) in enumerate(indexed):
                if (pid, threshold) not in matches:
                    continue  # outside COCO top100/category
                gid, ignored = matches[(pid, threshold)]
                if ignored:
                    continue
                same = [k for k, a in enumerate(anns) if a["category_id"] == p["category_id"]]
                same_best = max(same, key=lambda k: matrix[j, k]) if same else None
                best = int(matrix[j].argmax()) if anns and matrix[j].max() > 0 else None
                same_iou = float(matrix[j, same_best]) if same_best is not None else 0.
                any_iou = float(matrix[j, best]) if best is not None else 0.
                occupied = any(matrix[j, k] >= threshold and anns[k]["id"] in matched and matched[anns[k]["id"]] != pid for k in same)
                status = "TP" if gid else "FP"
                kind = "TP" if gid else fp_kind(same_iou, any_iou, occupied, threshold)
                associated = next((a for a in anns if a["id"] == gid), None) if gid else (
                    anns[same_best] if kind in ("Loc", "Dupe", "Unresolved") and same_best is not None else anns[best] if best is not None else None)
                area = associated.get("area", np.prod(associated["bbox"][2:])) if associated is not None and kind != "Bkg" else np.prod(p["bbox"][2:])
                review = reviews.get(pid, {})
                detections.append({"iou_threshold": threshold, "prediction_id": pid, "image_id": iid,
                    "category_id": p["category_id"], "score": p["score"], "bbox": json.dumps(p["bbox"]),
                    "status": status, "error": kind, "max_iou_same": same_iou, "max_iou_any": any_iou,
                    "matched_gt_id": int(gid), "associated_gt_id": associated["id"] if associated else "",
                    "associated_gt_category": associated["category_id"] if associated else "",
                    "size": base.size(float(area)), "size_basis": "GT" if associated is not None and kind != "Bkg" else "prediction",
                    "review_status": review.get("review_status", "pending"), "reviewer": review.get("reviewer", ""),
                    "evidence": review.get("evidence", ""), **scenes[iid]})
            for k, a in enumerate(anns):
                same = [j for j, p in enumerate(ps) if p["category_id"] == a["category_id"]]
                best = max(same, key=lambda j: matrix[j, k]) if same else None
                truths.append({"iou_threshold": threshold, "gt_id": a["id"], "image_id": iid,
                    "category_id": a["category_id"], "bbox": json.dumps(a["bbox"]),
                    "size": base.size(a.get("area", np.prod(a["bbox"][2:]))), "miss_all_scores": a["id"] not in matched,
                    "miss_score_ge_0_5": a["id"] not in matched_high,
                    "best_same_prediction_id": indexed[best][0] if best is not None else "", **scenes[iid]})
    return detections, truths


def counts(ds, gs):
    result = {"GT": len(gs), "evaluated_predictions": len(ds), "TP": sum(d["status"] == "TP" for d in ds),
              "FP": sum(d["status"] == "FP" for d in ds), "FN": sum(g["miss"] for g in gs)}
    result.update({k: sum(d["error"] == k for d in ds) for k in KINDS})
    result.update({"suspected_missing_annotation": sum(d["status"] == "FP" and d["review_status"] == "suspected_missing_annotation" for d in ds),
        "confirmed_missing_annotation": sum(d["status"] == "FP" and d["review_status"] == "confirmed_missing_annotation" for d in ds),
        "review_confirmed_background": sum(d["status"] == "FP" and d["error"] == "Bkg" and d["review_status"] == "confirmed_background" for d in ds),
        "Bkg_pending_or_uncertain": sum(d["error"] == "Bkg" and d["review_status"] in ("pending", "uncertain") for d in ds)})
    assert result["TP"]+result["FP"] == result["evaluated_predictions"]
    assert result["TP"]+result["FN"] == result["GT"]
    assert sum(result[k] for k in KINDS) == result["FP"]
    return result


def summarize(ds, gs, gt, cats, out):
    class_rows, size_rows, image_rows, distributions, histograms = [], [], [], [], []
    for threshold in (.5, .75):
        for scope in ("all_scores", "score_ge_0_5"):
            ds0 = [d for d in ds if d["iou_threshold"] == threshold and (scope == "all_scores" or d["score"] >= .5)]
            gs0 = [dict(g, miss=g["miss_all_scores"] if scope == "all_scores" else g["miss_score_ge_0_5"]) for g in gs if g["iou_threshold"] == threshold]
            dc, gc = defaultdict(list), defaultdict(list)
            for d in ds0: dc[d["category_id"]].append(d)
            for g in gs0: gc[g["category_id"]].append(g)
            for cid, name in cats.items():
                header = {"iou_threshold": threshold, "score_scope": scope, "category_id": cid, "category_name": name}
                class_rows.append(dict(header, **counts(dc[cid], gc[cid])))
                for size in ("small", "medium", "large"):
                    size_rows.append(dict(header, size=size, **counts_by_size(dc[cid], gc[cid], size)))
                pool = [d for d in dc[cid] if d["status"] == "FP"]
                for kind in ("all_FP",)+KINDS:
                    selected = [d for d in pool if kind == "all_FP" or d["error"] == kind]
                    summary = dict(header, error=kind, count=len(selected))
                    for field in ("score", "max_iou_same", "max_iou_any"):
                        vals = [d[field] for d in selected]
                        for label, fn in (("mean", np.mean), ("median", np.median), ("P90", lambda v: np.quantile(v, .9))):
                            summary[field+"_"+label] = float(fn(vals)) if vals else None
                        edges = (0., .1, .25, .5, .75, .9, 1.) if field == "score" else (0., .1, .25, .5, .75, 1.)
                        for n, (lo, hi) in enumerate(zip(edges, edges[1:])):
                            histograms.append(dict(header, error=kind, field=field, lower=lo, upper=hi,
                                upper_inclusive=n == len(edges)-2, count=sum(lo <= v <= hi if n == len(edges)-2 else lo <= v < hi for v in vals)))
                    distributions.append(summary)
                di, gi = defaultdict(list), defaultdict(list)
                for d in dc[cid]: di[d["image_id"]].append(d)
                for g in gc[cid]: gi[g["image_id"]].append(g)
                for image in gt["images"]:
                    image_rows.append(dict(header, image_id=image["id"], file_name=image["file_name"], **counts(di[image["id"]], gi[image["id"]])))
    for name, rows in (("class_statistics", class_rows), ("class_size_statistics", size_rows),
                       ("image_statistics", image_rows), ("fp_distribution_summary", distributions), ("fp_histograms", histograms)):
        base.write_csv(out/(name+".csv"), rows)
    return class_rows, image_rows


def counts_by_size(ds, gs, size):
    # FP size is explicitly an associated-GT size or prediction size. Thus TP+FN
    # still equals this size's GT count; FP is not divided by GT exposure.
    return counts([d for d in ds if d["size"] == size], [g for g in gs if g["size"] == size])


def scene_summary(image_rows, scenes, out):
    rows = []
    for key in ("scene_id", "scene_type"):
        groups = defaultdict(list)
        for r in image_rows:
            value = scenes[r["image_id"]][key]
            if value:
                groups[(r["iou_threshold"], r["score_scope"], r["category_id"], r["category_name"], value)].append(r)
        for (threshold, scope, cid, name, value), group in sorted(groups.items()):
            row = {"group_basis": key, "group": value, "iou_threshold": threshold, "score_scope": scope,
                   "category_id": cid, "category_name": name, "images": len(group)}
            for field in ("GT", "evaluated_predictions", "TP", "FP", "FN")+KINDS:
                row[field] = sum(r[field] for r in group)
            row["FP_per_image"] = row["FP"]/len(group)
            row["FN_fraction_GT"] = row["FN"]/row["GT"] if row["GT"] else None
            rows.append(row)
    base.write_csv(out/"scene_statistics.csv", rows, list(rows[0]) if rows else ["group_basis", "group", "images", "FP", "FN"])
    return rows


def visualize(ds, gs, gt, predictions, images, cats, image_root, out, prediction_sha):
    foreground = [d for d in ds if d["iou_threshold"] == .5 and d["score"] >= .5 and d["status"] == "FP"]
    pools = {k: [d for d in foreground if d["error"] == k] for k in KINDS}
    pools["FN"] = [g for g in gs if g["iou_threshold"] == .5 and g["miss_score_ge_0_5"]]
    by_gt = defaultdict(list)
    for a in gt["annotations"]: by_gt[a["image_id"]].append(a)
    index, review_templates = [], []
    for kind, pool in pools.items():
        pool.sort(key=lambda r: (-r.get("score", 0), r["image_id"], r.get("prediction_id", r.get("gt_id"))))
        selected = base.select_examples(pool, lambda r: (r["category_id"], r["size"]))  # at most20/type; category/size rotation
        folder = out/"visualizations"/kind
        folder.mkdir(parents=True, exist_ok=True)
        for n, row in enumerate(selected, 1):
            iid = row["image_id"]
            with Image.open(Path(image_root)/images[iid]["file_name"]) as original:
                if original.size != (images[iid]["width"], images[iid]["height"]):
                    raise ValueError(f"Original image dimensions differ: {iid}")
                canvas = original.convert("RGB")
            draw = ImageDraw.Draw(canvas)
            for a in by_gt[iid]:
                x, y, w, h = a["bbox"]
                draw.rectangle((x, y, x+w, y+h), outline="lime", width=2)
            focus = json.loads(row["bbox"])
            x, y, w, h = focus
            draw.rectangle((x, y, x+w, y+h), outline="orange" if kind == "FN" else "red", width=3)
            if kind == "FN":
                caption = f"FN IoU=.50 score>=.5 GT={row['gt_id']} (coverage != TP)"
            else:
                caption = f"{kind} pid={row['prediction_id']} s={row['score']:.3f} IoU_any={row['max_iou_any']:.3f} review={row['review_status']}"
            draw.text((4, 4), f"image_id={iid} category={row['category_id']}\n"+caption, fill="white", stroke_width=1, stroke_fill="black")
            filename = f"{n:02d}_image{iid}.png"
            canvas.save(folder/filename)
            pad = max(w, h, 32.)
            bounds = (max(0, math.floor(x-pad)), max(0, math.floor(y-pad)), min(canvas.width, math.ceil(x+w+pad)), min(canvas.height, math.ceil(y+h+pad)))
            if bounds[2] <= bounds[0] or bounds[3] <= bounds[1]: bounds = (0, 0, canvas.width, canvas.height)
            crop = canvas.crop(bounds)
            factor = 640/max(crop.size)
            crop.resize((max(1, round(crop.width*factor)), max(1, round(crop.height*factor))), Image.Resampling.NEAREST).save(folder/filename.replace(".png", "_crop.png"))
            index.append({"kind": kind, "image_id": iid, "category_id": row["category_id"], "category_name": cats[row["category_id"]],
                "prediction_id": row.get("prediction_id", ""), "gt_id": row.get("gt_id", ""),
                "file": str(Path("visualizations")/kind/filename), "record": json.dumps(row)})
            if kind == "Bkg":
                review_templates.append({"prediction_id": row["prediction_id"], "image_id": iid, "prediction_sha256": prediction_sha,
                    "review_status": row["review_status"], "reviewer": row["reviewer"], "evidence": row["evidence"]})
    base.write_csv(out/"visualization_index.csv", index, list(index[0]) if index else ["kind", "image_id", "prediction_id", "gt_id", "file", "record"])
    base.write_csv(out/"background_review.csv", review_templates, ["prediction_id", "image_id", "prediction_sha256", "review_status", "reviewer", "evidence"])
    return Counter(r["kind"] for r in index)


def report(class_rows, image_rows, scene_rows, scene_coverage, example_counts, manifest, out):
    primary = [r for r in class_rows if r["iou_threshold"] == .5 and r["score_scope"] == "score_ge_0_5"]
    lines = ["# ERROR_STRUCTURE_REPORT", "", "## 固定口径与来源", "",
        "只读完整既有预测/GT；不构建模型、不推理、不训练。主统计score≥0.5、IoU=.50，原COCO一对一匹配，maxDets=100/图/类别。另列IoU=.75及全分数结果，不做阈值sweep。crowd/ignored与超maxDets检测不计FP。",
        "FP是相对于现有标注协议的FP，Bkg是与全部已标注非crowd GT的IoU<.1；未经复核不称为真实背景/杂波。",
        f"预测SHA256：`{manifest['prediction_sha256']}`；GT副本SHA256：`{manifest['GT_copy_sha256']}`。来源及限制见analysis_manifest.json。",
        f"旧GT来源记录核对：{manifest['GT_source_verified']}。来源文件缺失且副本字节SHA不能直接链接时，该项保留未验证，不当作来源已证明。",
        f"baseline导出来源链接：{manifest['export_link_verified']}；缺少链接时不把目录名当checkpoint证明。", "",
        "## 各类别错误结构", "", "|类别|GT|TP|FP|FN|标注Bkg|Cls|Loc|Dupe|Both|Unresolved|", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in primary:
        lines.append("| "+" | ".join(str(r[k]) for k in ("category_name", "GT", "TP", "FP", "FN")+KINDS)+" |")
    lines += ["", "Cls：强异类重叠且无强同类重叠；Loc：同类IoU∈[.1,匹配阈值)且无强异类重叠；Dupe：强同类重叠的GT已被其它预测匹配；Both：弱异类重叠、同类IoU<.1；无法确认重复依据时单列Unresolved。这是标注相对的结构规则，不是TIDE dAP，也不证明真实目标类别或物理机制。", "",
        "## 置信度、尺度与集中程度", "",
        "fp_distribution_summary.csv保留各类FP的score/同类IoU/任意类别IoU均值、中位数、P90；fp_histograms.csv为固定分箱计数。all_scores与score_ge_0_5不混用。",
        "class_size_statistics.csv：TP/FN用GT area，Bkg用预测面积，其余FP使用关联GT面积（缺少关联则预测面积）；details表逐框标记size_basis。small<32²、medium<96²、large≥96²采用原图面积，不能当实际640输入短边。", ""]
    for r in primary:
        rows = [i for i in image_rows if i["iou_threshold"] == .5 and i["score_scope"] == "score_ge_0_5" and i["category_id"] == r["category_id"]]
        n = max(1, math.ceil(len(rows)*.1))
        fp_top = sum(sorted((i["FP"] for i in rows), reverse=True)[:n])
        fn_top = sum(sorted((i["FN"] for i in rows), reverse=True)[:n])
        lines.append(f"- {r['category_name']}：错误最多的前10%图像分别承载FP {fp_top}/{r['FP']}、FN {fn_top}/{r['FN']}。FP与FN分别排序，0分母为无此错误；需要同时查看逐图GT/评价预测暴露量，不能据此声称某场景固有困难。")
    lines += ["", f"明确scene_id覆盖{scene_coverage['scene_id']}/{scene_coverage['images']}图，scene_type覆盖{scene_coverage['scene_type']}/{scene_coverage['images']}图。",
        "scene_statistics.csv仅对明确标签分组并给出图数、GT数、FP/图、FN/GT。缺失标签不靠文件名/类别猜；逐图集中不等于原始场景集中，切片还需scene_id映射。", "",
        "## 漏标复核与案例", ""]
    for r in primary:
        lines.append(f"- {r['category_name']}：疑似漏标记录{r['suspected_missing_annotation']}，已确认漏标记录{r['confirmed_missing_annotation']}；确认背景Bkg {r['review_confirmed_background']}，Bkg待复核/不确定 {r['Bkg_pending_or_uncertain']}。0条漏标记录只表示没有相应复核记录，不代表无漏标。")
    lines += ["", f"案例数量：{dict(example_counts)}；每错误类型最多20张，优先高score、类别/尺度轮转，同类型不重复image_id，不复制补足。Bkg案例优先供人工排查漏标。",
        "background_review.csv为复核表，人工填写状态、审阅者、具体证据；用新输出目录与--reviews重读同一预测即可合并，不修改GT/正式COCO AP。疑似/确认漏标在证据统计中单列，仍保留标注协议下原FP。", "",
        "## 有证据支持与不能归因", "",
        "上述TP/FP/FN、score/IoU、尺度与明确标签分组支持描述性结论。它们不能直接证明SAR散射、斑点、成像几何、边界模糊或目标—杂波混淆机制。复核不足的Bkg可能包含漏标；Loc仅说明标注相对定位不足；Both也不确认物体真实类别。",
        "场景标签缺失时，场景类型结论未评估；未人工看原图不能归因于漏标或SAR机制。本自动报告没有声称完成视觉机制审阅。", "",
        "输出：class_statistics.csv、class_size_statistics.csv、image_statistics.csv、scene_statistics.csv、fp_distribution_summary.csv、fp_histograms.csv、high_confidence_detection_details.csv、gt_detection_details.csv、background_review.csv、visualizations/、visualization_index.csv、coco_metrics.json、analysis_manifest.json、COMPLETE.json。"]
    (out/"ERROR_STRUCTURE_REPORT.md").write_text("\n".join(lines)+"\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", required=True, help="Existing full error-analysis directory")
    parser.add_argument("--images", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--export-dir", help="Optional original E0 export: metadata plus complete predictions.json")
    parser.add_argument("--scene-map", help="Optional CSV: image_id,scene_id,scene_type; partial coverage allowed")
    parser.add_argument("--reviews", help="Optional completed background_review.csv linked to original prediction SHA")
    args = parser.parse_args()
    source, out = Path(args.analysis), Path(args.output)
    if out.exists() and any(p.name not in ("console.log", "launcher.pid") for p in out.iterdir()):
        raise FileExistsError("Use a fresh error-structure output directory")
    gt_path, pred_path = source/"validation_gt.json", source/"predictions.json"
    gt, predictions, previous = load(gt_path), load(pred_path), load(source/"analysis_manifest.json")
    pred_sha = base.sha(pred_path)
    if pred_sha != previous["predictions_sha256"]:
        raise ValueError("Existing predictions do not match analysis manifest")
    original_gt = Path(previous["annotations_source"])
    gt_source_verified = False
    if original_gt.is_file():
        if base.sha(original_gt) != previous["annotations_sha256"] or load(original_gt) != gt:
            raise ValueError("Recorded GT source changed or differs from analysis GT")
        gt_source_verified = True
    elif base.sha(gt_path) == previous["annotations_sha256"]:
        gt_source_verified = True
    images, cats = base.preflight(gt, predictions, args.images)
    scenes = scene_map(gt, args.scene_map)
    reviews = review_map(args.reviews, predictions, pred_sha)
    if set(reviews)-{pid for pid, p in enumerate(predictions) if p["score"] >= .5}:
        raise ValueError("Reviews must concern score>=.5 detections")
    export_link, export_metadata = False, None
    if args.export_dir:
        export = Path(args.export_dir)
        export_metadata = load(export/"export_metadata.json")
        if base.sha(export/"predictions.json") != pred_sha or not export_metadata["strict_load"]:
            raise ValueError("Original export/prediction link differs")
        dec = export_metadata["resolved_config"].get("RTDETRTransformerv2", {})
        if any(dec.get(k, False) for k in ("sar_quality", "decoder_quality", "umqr", "sber", "final_quality_probe",
                "class_conditioned_final_quality_probe", "multi_threshold_class_conditioned_quality_probe")):
            raise ValueError("Requires E0 baseline export")
        if dec.get("query_select_method", "default") != "default":
            raise ValueError("Requires baseline query selection")
        post = export_metadata["resolved_config"].get("RTDETRPostProcessor", {})
        if post.get("final_score_method", "default") != "default" or any(post.get(k, 0) != 0 for k in (
                "final_quality_gamma", "oracle_final_iou_gamma", "class_aware_oracle_final_iou_gamma", "pairwise_class_aware_oracle_beta")):
            raise ValueError("Requires baseline postprocessor without oracle/quality reranking")
        if "image_count" in export_metadata and export_metadata["image_count"] != len(images):
            raise ValueError("Original export validation image count differs")
        if "num_top_queries" in post:
            per_image = Counter(p["image_id"] for p in predictions)
            if set(per_image) != set(images) or set(per_image.values()) != {post["num_top_queries"]}:
                raise ValueError("Input is not the complete original top-query export")
        export_link = True
    out.mkdir(parents=True, exist_ok=True)
    print("Sources validated; reusing COCO evaluator, no inference", flush=True)
    metrics, matches, _ = base.coco_evaluate(gt_path, predictions, out)  # no TIDE dependency/check/model import
    ds, gs = records(gt, predictions, matches, scenes, reviews)
    fields = ["iou_threshold", "prediction_id", "image_id", "category_id", "score", "bbox", "status", "error", "max_iou_same", "max_iou_any", "matched_gt_id", "associated_gt_id", "associated_gt_category", "size", "size_basis", "review_status", "reviewer", "evidence", "scene_id", "scene_type"]
    base.write_csv(out/"high_confidence_detection_details.csv", [d for d in ds if d["score"] >= .5], fields)
    base.write_csv(out/"gt_detection_details.csv", gs, ["iou_threshold", "gt_id", "image_id", "category_id", "bbox", "size", "miss_all_scores", "miss_score_ge_0_5", "best_same_prediction_id", "scene_id", "scene_type"])
    classes, per_image = summarize(ds, gs, gt, cats, out)
    scene_rows = scene_summary(per_image, scenes, out)
    example_counts = visualize(ds, gs, gt, predictions, images, cats, args.images, out, pred_sha)
    coverage = {k: sum(bool(s[k]) for s in scenes.values()) for k in ("scene_id", "scene_type")}
    coverage["images"] = len(images)
    manifest = {"prediction_sha256": pred_sha, "GT_copy_sha256": base.sha(gt_path),
        "predictions_source": str(pred_path.resolve()), "annotations_source": str(gt_path.resolve()),
        "images_root": str(Path(args.images).resolve()),
        "recorded_GT_source_sha256": previous["annotations_sha256"], "GT_source_verified": gt_source_verified,
        "source_analysis": str(source), "source_analysis_manifest_sha256": base.sha(source/"analysis_manifest.json"),
        "source_analysis_git_commit": previous.get("git_commit"), "export_link_verified": export_link,
        "export_metadata": export_metadata, "checkpoint_bytes_independently_checked": False,
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=base.ROOT, text=True).strip(),
        "script_sha256": base.sha(__file__),
        "scene_map_sha256": base.sha(args.scene_map) if args.scene_map else None,
        "review_csv_sha256": base.sha(args.reviews) if args.reviews else None, "scene_coverage": coverage,
        "scope": "one CPU offline pass; no model/inference/training; FP labels are annotation-relative; no SAR mechanism attribution"}
    base.dump(out/"analysis_manifest.json", manifest)
    report(classes, per_image, scene_rows, coverage, example_counts, manifest, out)
    base.dump(out/"COMPLETE.json", {"status": "complete", "visualizations": dict(example_counts), "scene_coverage": coverage,
        "visual_mechanism_review": "not_performed_by_script", "metrics": metrics})
    print(f"Complete: {out/'ERROR_STRUCTURE_REPORT.md'}", flush=True)


if __name__ == "__main__":
    main()
