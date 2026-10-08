"""Read-only SAR image-evidence diagnostics; never updates detector parameters.

Run --self-test first. Real inference requires an explicit config and checkpoint.
Image brightness is an observable proxy, not a measured scattering centre or SLC.
"""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))


def dependencies():
    global np, Image, yaml
    import numpy as np
    from PIL import Image
    import yaml


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def dump(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".partial")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    tmp.replace(path)


def write_csv(path, rows):
    fields = list(dict.fromkeys(k for row in rows for k in row)) or ["image_id"]
    with Path(path).open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def protocol(path):
    p = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not (1 < p["ring_guard_scale"] < p["ring_outer_scale"]):
        raise ValueError("ring scales must satisfy 1 < guard < outer")
    if p["primary_quantile"] not in p["evidence_quantiles"] or p["primary_score"] not in p["score_thresholds"]:
        raise ValueError("primary thresholds must be included in sensitivity thresholds")
    if any(not 0 < q < 1 for q in p["evidence_quantiles"]):
        raise ValueError("evidence quantiles must be between 0 and 1")
    if any(not 0 <= s <= 1 for s in p["score_thresholds"]):
        raise ValueError("scores must be between 0 and 1")
    if not 0 <= p["background_max_iou"] < p["match_iou"] <= p["precise_iou"] <= 1:
        raise ValueError("invalid IoU thresholds")
    for k in ("minimum_background_pixels", "minimum_foreground_pixels", "minimum_evidence_pixels",
              "bootstrap_repetitions", "minimum_association_objects", "minimum_association_images", "minimum_bin_objects"):
        if p[k] < 1:
            raise ValueError(f"{k} must be positive")
    if p["robust_noise_floor_fraction"] <= 0:
        raise ValueError("robust noise floor must be positive")
    return p


def xyxy(ann):
    x, y, w, h = ann["bbox"]
    return [x, y, x + w, y + h]


def iou_matrix(a, b, crowd=False):
    a, b = np.asarray(a, dtype=float).reshape(-1, 4), np.asarray(b, dtype=float).reshape(-1, 4)
    wh = np.maximum(0, np.minimum(a[:, None, 2:], b[None, :, 2:]) - np.maximum(a[:, None, :2], b[None, :, :2]))
    intersection = wh.prod(axis=2)
    area_a = np.maximum(0, a[:, 2:] - a[:, :2]).prod(axis=1)
    area_b = np.maximum(0, b[:, 2:] - b[:, :2]).prod(axis=1)
    denominator = area_a[:, None] if crowd else area_a[:, None] + area_b[None, :] - intersection
    return intersection / np.maximum(denominator, 1e-12)


def image_array(path):
    with Image.open(path) as im:
        array = np.asarray(im).astype(float)
    if array.ndim == 2:
        return array, "single_channel"
    if array.ndim != 3 or array.shape[2] < 3:
        raise ValueError(f"unsupported image channels: {path}")
    rgb = array[:, :, :3]
    mode = "replicated_gray" if np.max(np.ptp(rgb, axis=2)) == 0 else "RGB_luminance_proxy"
    return rgb @ np.array([.299, .587, .114]), mode


def evidence(gray, box, exclude_boxes, p, quantile):
    """Pixel-centre sampling, guarded ring, annotation exclusion (diagnostics only)."""
    H, W = gray.shape
    x1, y1, x2, y2 = box
    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(W, x2), min(H, y2)
    w, h = x2 - x1, y2 - y1
    if min(w, h) <= 0:
        return {"proxy_status": "outside_image"}
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    s, g = p["ring_outer_scale"], p["ring_guard_scale"]
    left, right = max(0, math.floor(cx - s*w/2)), min(W, math.ceil(cx + s*w/2))
    top, bottom = max(0, math.floor(cy - s*h/2)), min(H, math.ceil(cy + s*h/2))
    yy, xx = np.mgrid[top:bottom, left:right].astype(float)
    xx += .5; yy += .5
    inside = (xx >= x1) & (xx < x2) & (yy >= y1) & (yy < y2)
    ring = ((np.abs(xx - cx) >= g*w/2) | (np.abs(yy - cy) >= g*h/2)) & (np.abs(xx-cx) < s*w/2) & (np.abs(yy-cy) < s*h/2)
    available_ring = int(ring.sum())
    for bx1, by1, bx2, by2 in exclude_boxes:
        if bx2 > left and bx1 < right and by2 > top and by1 < bottom:
            ring &= ~((xx >= bx1) & (xx < bx2) & (yy >= by1) & (yy < by2))
    crop = gray[top:bottom, left:right]
    fg, bg = crop[inside], crop[ring]
    result = {"foreground_pixels": int(fg.size), "background_pixels": int(bg.size),
              "background_available_fraction": float(bg.size/max(1, available_ring)),
              "ring_clipped": bool(cx-s*w/2 < 0 or cy-s*h/2 < 0 or cx+s*w/2 > W or cy+s*h/2 > H)}
    if bg.size < p["minimum_background_pixels"] or fg.size < p["minimum_foreground_pixels"]:
        return dict(result, proxy_status="insufficient_pixels")
    median = float(np.median(bg))
    mad = 1.4826 * float(np.median(np.abs(bg - median)))
    bg_spread = float(np.quantile(bg, .99)-np.quantile(bg, .01))
    # The fallback scale depends on background only, never on the target's peak.
    scale = max(mad, p["robust_noise_floor_fraction"] * max(abs(median), bg_spread, 1), 1e-12)
    result.update({"background_median": median, "background_robust_scale": mad,
                   "background_tail_z": float((np.quantile(bg, .99)-median)/scale),
                   "foreground_contrast_z": float((np.median(fg)-median)/scale),
                   "foreground_peak_z": float((np.quantile(fg, .99)-median)/scale)})
    threshold = float(np.quantile(fg, quantile))
    weights = np.where(inside & (crop >= threshold), np.maximum(crop - median, 0), 0)
    positive = int((weights > 0).sum())
    result["evidence_pixels"] = positive
    if positive < p["minimum_evidence_pixels"] or weights.sum() <= 0:
        return dict(result, proxy_status="no_positive_bright_evidence")
    ex, ey = float((weights*xx).sum()/weights.sum()), float((weights*yy).sum()/weights.sum())
    dx, dy = ex-cx, ey-cy
    result.update({"proxy_status": "valid", "evidence_x": ex, "evidence_y": ey,
                   "evidence_dx": dx, "evidence_dy": dy,
                   "evidence_offset": float(math.hypot(dx, dy)/math.hypot(w, h)),
                   "evidence_fraction": positive/int(fg.size)})
    return result


def pair_predictions(data, stage, topk):
    boxes = data[f"{stage}_boxes_cxcywh"].astype(float)
    logits = data[f"{stage}_logits"].astype(float)
    if boxes.shape != (len(logits), 4) or logits.ndim != 2 or not np.isfinite(boxes).all() or not np.isfinite(logits).all():
        raise ValueError("invalid query arrays")
    scores = 1/(1+np.exp(-np.clip(logits, -700, 700)))
    wh = data["original_image_wh"]
    corners = np.concatenate((boxes[:, :2]-boxes[:, 2:]/2, boxes[:, :2]+boxes[:, 2:]/2), axis=1) * np.tile(wh, 2)
    # Use authoritative pair IDs exported from torch.topk; legacy imports are refused.
    pair_key = f"{stage}_actual_output_pair_ids"
    if pair_key not in data:
        raise ValueError("query file lacks exact output pair IDs; run this export tool rather than legacy sar_audit")
    pairs = data[pair_key].astype(int)
    if len(pairs) != min(topk, logits.size) or len(set(pairs.tolist())) != len(pairs) or np.any(pairs < 0) or np.any(pairs >= logits.size):
        raise ValueError("invalid output pair IDs")
    queries, labels = pairs//logits.shape[1], pairs % logits.shape[1]
    emitted_boxes = data[f"{stage}_actual_output_boxes_xyxy"].astype(float)
    emitted_scores = data[f"{stage}_actual_output_scores"].astype(float)
    if emitted_boxes.shape != (len(pairs), 4) or emitted_scores.shape != (len(pairs),):
        raise ValueError("invalid postprocessed output shape")
    if not np.allclose(emitted_scores, scores.ravel()[pairs], rtol=1e-5, atol=1e-7) or not np.allclose(emitted_boxes, corners[queries], rtol=1e-5, atol=1e-4):
        raise ValueError("postprocessed output differs from class-pair TopK")
    return corners, emitted_boxes, labels, emitted_scores, queries


def match_predictions(boxes, labels, scores, anns, p):
    """Score-greedy unique matching at fixed IoU; separate ignored, duplicate, class errors."""
    eligible = [a for a in anns if not a.get("iscrowd", 0) and not a.get("ignore", 0)]
    gt_boxes = [xyxy(a) for a in eligible]
    overlaps = iou_matrix(boxes, gt_boxes)
    gt_labels = np.array([a["category_id"] for a in eligible])
    used, matched, types = set(), {}, [None]*len(boxes)
    ignored = [a for a in anns if a.get("iscrowd", 0) or a.get("ignore", 0)]
    ignore_hit = np.zeros(len(boxes), dtype=bool)
    for a in ignored:
        ignore_hit |= iou_matrix(boxes, [xyxy(a)], crowd=bool(a.get("iscrowd", 0)))[:, 0] >= p["match_iou"]
    for j in np.argsort(-scores, kind="stable"):
        candidates = [k for k in range(len(eligible)) if gt_labels[k] == labels[j] and k not in used and overlaps[j, k] >= p["match_iou"]]
        if candidates:
            k = max(candidates, key=lambda k: overlaps[j, k])
            used.add(k); matched[k] = int(j); types[j] = "TP"
        elif ignore_hit[j]:
            types[j] = "ignored_region"
        elif len(eligible) and np.any((gt_labels == labels[j]) & (overlaps[j] >= p["match_iou"])):
            types[j] = "duplicate"
        elif len(eligible) and np.any(overlaps[j] >= p["match_iou"]):
            types[j] = "class_error"
        elif len(eligible) and overlaps[j].max() >= p["background_max_iou"]:
            types[j] = "localization_or_partial_overlap"
        else:
            types[j] = "background_candidate"
    return eligible, overlaps, matched, types


def geometry(pred, gt, proxy):
    pc, gc = (pred[:2]+pred[2:])/2, (gt[:2]+gt[2:])/2
    wh, pwh = gt[2:]-gt[:2], pred[2:]-pred[:2]
    delta = pc-gc
    result = {"center_error": float(np.linalg.norm(delta)/np.linalg.norm(wh)),
              "center_dx": float(delta[0]), "center_dy": float(delta[1]),
              "log_area_ratio": float(np.log(max(float(pwh.prod()), 1e-12)/float(wh.prod())))}
    if proxy.get("proxy_status") == "valid":
        ed = np.array([proxy["evidence_dx"], proxy["evidence_dy"]])
        norm = np.linalg.norm(ed)
        result["shift_toward_evidence"] = float(delta @ ed/(norm*np.linalg.norm(wh))) if norm > 1e-9 else None
    return result


def analyze_image(image, anns, data, gray, mode, p, topk):
    if gray.shape != (image["height"], image["width"]) or list(data["original_image_wh"]) != [image["width"], image["height"]]:
        raise ValueError(f"image dimensions differ from metadata: {image['id']}")
    stages = data["stage_names"].tolist()
    all_final, boxes, labels, scores, queries = pair_predictions(data, stages[-1], topk)
    all_encoder, _, _, _, _ = pair_predictions(data, stages[0], topk)
    if stages[0] != "encoder":
        raise ValueError("encoder stage missing")
    primary = scores >= p["primary_score"]
    eligible, _, matched, _ = match_predictions(boxes[primary], labels[primary], scores[primary], anns, p)
    ids = [a["id"] for a in eligible]
    if ids != data["GT_annotation_ids"].tolist() or not np.allclose(np.array([xyxy(a) for a in eligible]).reshape(-1, 4), data["GT_boxes_xyxy"]):
        raise ValueError("GT differs from query export; invalid/clipped GT require explicit review")
    gts = np.array([xyxy(a) for a in eligible]).reshape(-1, 4)
    enc_ious, final_ious = iou_matrix(all_encoder, gts), iou_matrix(all_final, gts)
    primary_ious = iou_matrix(boxes[primary], gts)
    exclusions = [xyxy(a) for a in anns]
    source_value = image.get(p["source_field"]) if p["source_field"] else None
    source = "unknown" if source_value is None else str(source_value)
    gt_rows, pred_rows = [], []
    for k, a in enumerate(eligible):
        gt = gts[k]
        w, h = gt[2:]-gt[:2]
        best = int(final_ious[:, k].argmax())
        enc_max, final_max = float(enc_ious[:, k].max()), float(final_ious[best, k])
        individual = bool(np.any((labels[primary] == a["category_id"]) & (primary_ious[:, k] >= p["match_iou"])))
        j = matched.get(k)
        matched_iou = float(primary_ious[j, k]) if j is not None else None
        if j is not None:
            failure = "detected_precise" if matched_iou >= p["precise_iou"] else "detected_low_precision"
        elif enc_max < p["match_iou"]:
            failure = "encoder_geometry_not_covered"
        elif final_max < p["match_iou"]:
            failure = "final_geometry_not_covered"
        elif not individual:
            failure = "class_score_or_output_selection"
        else:
            failure = "unique_matching_competition"
        for q in p["evidence_quantiles"]:
            proxy = evidence(gray, gt, exclusions, p, q)
            row = {"image_id": image["id"], "annotation_id": a["id"], "file_name": image["file_name"],
                   "category_id": a["category_id"], "source": source, "image_mode": mode,
                   "quantile": q, "area": float(w*h), "aspect": float(w/h),
                   "size": "small" if w*h < 1024 else "medium" if w*h < 9216 else "large",
                   "encoder_best_iou": enc_max, "final_best_iou": final_max,
                   "oracle_iou_loss": 1-final_max, "missed": int(j is None), "failure": failure,
                   "matched_iou": matched_iou, **proxy}
            # All-query geometry is an optimistic ceiling; not a detection score.
            if final_max > 0:
                row.update({f"oracle_{name}": val for name, val in geometry(all_final[best], gt, proxy).items()})
            if j is not None:
                row.update({f"matched_{name}": val for name, val in geometry(boxes[primary][j], gt, proxy).items()})
            gt_rows.append(row)
    cache = {}
    for threshold in p["score_thresholds"]:
        keep = np.flatnonzero(scores >= threshold)
        _, _, _, kinds = match_predictions(boxes[keep], labels[keep], scores[keep], anns, p)
        for j, kind in zip(keep, kinds):
            query = int(queries[j])
            if query not in cache:
                cache[query] = evidence(gray, boxes[j], exclusions, p, p["primary_quantile"])
            b = boxes[j]; w, h = b[2:]-b[:2]
            pred_rows.append({"image_id": image["id"], "source": source, "category_id": int(labels[j]),
                              "query_id": int(queries[j]), "score_threshold": threshold, "score": float(scores[j]),
                              "kind": kind, "area": float(w*h), "aspect": float(w/max(h, 1e-12)), **cache[query]})
    return gt_rows, pred_rows


def association(rows, xname, outcomes, p):
    """Partial rank correlation, nuisance-adjusted; image cluster bootstrap with refitting."""
    from scipy.stats import rankdata
    rows = [r for r in rows if all(r.get(k) is not None and math.isfinite(float(r[k])) for k in [xname, "area", "aspect", *outcomes]) and r["area"] > 0 and r["aspect"] > 0]
    ids = sorted({r["image_id"] for r in rows})
    result = {"objects": len(rows), "images": len(ids), "controls": ["class", "log_area_rank", "log_aspect_rank", "source_if_available"],
              "source_available": any(r["source"] != "unknown" for r in rows), "outcomes": {}}
    if len(rows) < p["minimum_association_objects"] or len(ids) < p["minimum_association_images"]:
        return dict(result, status="insufficient_samples")
    # Category/source fixed effects and within-category size/aspect slopes.
    continuous = np.column_stack([rankdata([math.log(r[k]) for r in rows]) for k in ("area", "aspect")])
    continuous = (continuous-continuous.mean(0))/np.maximum(continuous.std(0), 1e-12)
    columns = [np.ones(len(rows)), *continuous.T]
    for key in ("category_id", "source"):
        values = sorted({r[key] for r in rows})
        for value in values[1:]:
            dummy = np.array([r[key] == value for r in rows], dtype=float)
            columns.append(dummy)
            if key == "category_id":
                columns.extend((dummy*continuous[:, 0], dummy*continuous[:, 1]))
    X = np.column_stack(columns)
    result["control_design_rank"] = int(np.linalg.matrix_rank(X))
    if len(rows) < result["control_design_rank"] + 10:
        return dict(result, status="insufficient_residual_degrees_of_freedom")
    Y = np.column_stack([rankdata([r[k] for r in rows]) for k in [xname, *outcomes]])
    def fit(weights):
        sw = np.sqrt(weights)
        residual = Y-X @ np.linalg.lstsq(X*sw[:, None], Y*sw[:, None], rcond=None)[0]
        residual -= np.average(residual, weights=weights, axis=0)
        var = (weights[:, None]*residual**2).sum(0)
        cov = (weights[:, None]*residual[:, :1]*residual[:, 1:]).sum(0)
        return np.divide(cov, np.sqrt(var[:1]*var[1:]), out=np.full(len(outcomes), np.nan), where=np.sqrt(var[:1]*var[1:]) > 1e-9)
    rho = fit(np.ones(len(rows)))
    lookup = {v: i for i, v in enumerate(ids)}
    clusters = np.array([lookup[r["image_id"]] for r in rows])
    rng = np.random.default_rng(p["seed"])
    boot = np.array([fit(np.bincount(rng.integers(len(ids), size=len(ids)), minlength=len(ids))[clusters]) for _ in range(p["bootstrap_repetitions"])])
    for i, name in enumerate(outcomes):
        valid = boot[:, i][np.isfinite(boot[:, i])]
        result["outcomes"][name] = {"partial_rank_rho": float(rho[i]) if np.isfinite(rho[i]) else None,
                                    "image_bootstrap_CI95": np.quantile(valid, [.025, .975]).tolist() if len(valid) else None,
                                    "valid_bootstrap_repetitions": int(len(valid))}
    return dict(result, status="descriptive_association_NOT_causal")


def summarize(gt, predictions, p):
    primary = [r for r in gt if r["quantile"] == p["primary_quantile"]]
    result = {"GT_count": len(primary), "proxy_status_counts": dict(Counter(r["proxy_status"] for r in primary)),
              "failure_counts": dict(Counter(r["failure"] for r in primary)), "evidence_associations": {}, "background_associations": {},
              "limitations": ["Brightness proxy is not a physical scattering-centre measurement; PNG radiometry may be transformed.",
                              "Association does not establish causality or guarantee a module improvement.",
                              "All-query best IoU is a geometry upper bound, not actual detector recall.",
                              "FP types use fixed-threshold diagnostic matching; official AP is stored separately.",
                              "Background candidates may contain unlabeled objects; ring statistics exclude known GT only.",
                              "Matched-only localisation is subject to detection selection bias.",
                              "Source is unknown unless real source metadata is explicitly supplied; source robustness then remains unverified.",
                              "Sensitivity analyses are descriptive; do not select a quantile by its most favourable result."]}
    for q in p["evidence_quantiles"]:
        rows = [r for r in gt if r["quantile"] == q and r["proxy_status"] == "valid"]
        result["evidence_associations"][str(q)] = {
            "all_GT": association(rows, "evidence_offset", ["oracle_iou_loss", "missed"], p),
            "overlapping_oracle_queries": association(rows, "evidence_offset", ["oracle_center_error", "oracle_log_area_ratio"], p),
            "oracle_direction_defined": association(rows, "evidence_offset", ["oracle_shift_toward_evidence"], p),
            "matched_only": association(rows, "evidence_offset", ["matched_center_error", "matched_log_area_ratio"], p),
            "matched_direction_defined": association(rows, "evidence_offset", ["matched_shift_toward_evidence"], p)}
    for score in p["score_thresholds"]:
        rows = [r for r in predictions if r["score_threshold"] == score]
        comparison = [dict(r, background_FP=int(r["kind"] == "background_candidate")) for r in rows if r["kind"] in ("TP", "background_candidate")]
        result["background_associations"][str(score)] = {"prediction_types": dict(Counter(r["kind"] for r in rows)),
            "background_FP_vs_TP": association(comparison, "background_tail_z", ["background_FP"], p)}
    bins = []
    # Quantiles formed within class/COCO size strata; sparse strata are explicitly excluded.
    buckets = defaultdict(list)
    for r in primary:
        if r["proxy_status"] == "valid":
            buckets[(r["category_id"], r["size"], r["source"])].append(r)
    for (category, size, source), rows in sorted(buckets.items()):
        ordered = sorted(rows, key=lambda r: (r["evidence_offset"], r["annotation_id"]))
        if len(rows) < 3*p["minimum_bin_objects"]:
            bins.append({"category_id": category, "size": size, "source": source, "status": "insufficient_samples", "count": len(rows)})
            continue
        edges = np.quantile([r["evidence_offset"] for r in ordered], [1/3, 2/3])
        for i in range(3):
            group = [r for r in ordered if int(np.searchsorted(edges, r["evidence_offset"], side="right")) == i]
            if not group:
                continue
            bins.append({"category_id": category, "size": size, "source": source, "bin": i+1,
                         "status": "descriptive" if len(group) >= p["minimum_bin_objects"] else "insufficient_samples",
                         "count": len(group), "offset_median": float(np.median([r["evidence_offset"] for r in group])),
                         "oracle_iou_mean": float(np.mean([r["final_best_iou"] for r in group])),
                         "miss_rate": float(np.mean([r["missed"] for r in group]))})
    result["stratified_bins"] = bins
    return result


def report(out, result, p, sample_set):
    lines = ["# SAR 图像证据与检测失效诊断", "", f"样本范围：`{sample_set}`。主证据分位数 {p['primary_quantile']}；主分数阈值 {p['primary_score']}。",
             "", f"GT 数量：{result['GT_count']}。有效性：`{json.dumps(result['proxy_status_counts'], ensure_ascii=False)}`。",
             "", "这里的亮度证据只描述当前图像的可观测特征，不能直接称为物理散射中心。", "",
             "## 失效分布", "", "| 类型 | GT 数量 |", "|---|---:|"]
    lines += [f"| {name} | {count} |" for name, count in result["failure_counts"].items()]
    lines += ["", "## 证据偏移关联", "", "控制类别、面积、长宽比；传感器信息仅在真实 metadata 存在时控制。CI 按整幅图像聚类重采样。",
              "", "| 分位数 | 对象范围 | 指标 | 偏相关 | 95% CI | GT/图像 |", "|---|---|---|---:|---|---|"]
    for q, groups in result["evidence_associations"].items():
        for group, analysis in groups.items():
            if not analysis["outcomes"]:
                lines.append(f"| {q} | {group} | 样本不足 | — | — | {analysis['objects']}/{analysis['images']} |")
            for name, value in analysis["outcomes"].items():
                rho = value["partial_rank_rho"]
                ci = value["image_bootstrap_CI95"]
                formatted = f"{rho:.3f}" if rho is not None else "不可估计"
                lines.append(f"| {q} | {group} | {name} | {formatted} | {ci} | {analysis['objects']}/{analysis['images']} |")
    lines += ["", "## 局部背景关联", "", "`background_tail_z` 是背景环的亮尾统计，使用局部 MAD 标准化；不代表校准后的雷达 SCR。",
              "", "| 分数阈值 | 预测类型数量 | 背景 FP / TP 偏相关 | 95% CI |", "|---|---|---:|---|"]
    for score, group in result["background_associations"].items():
        a = group["background_FP_vs_TP"]["outcomes"].get("background_FP", {})
        lines.append(f"| {score} | {group['prediction_types']} | {a.get('partial_rank_rho', '样本不足')} | {a.get('image_bootstrap_CI95', '—')} |")
    lines += ["", "## 判断标准", "", "先检查主阈值的方向和置信区间，再检查两个备用分位数是否方向一致。全 GT 的结果、漏检率、匹配成功后的定位误差需要一起解释。",
              "证据偏移若只与小目标或某一类有关，或控制变量后不稳定，就不足以支持通用的几何校正模块。",
              "若 encoder 已缺乏几何覆盖，优先研究候选构建；若候选覆盖充足而最终定位变差，再研究证据聚合与框回归。",
              "背景统计关联仅提供下一步实验依据；要用针对背景条件的干预和消融验证机制。",
              "若数据没有来源标签，跨传感器的结论暂不成立。", "", "## 限制", ""]
    lines += [f"- {item}" for item in result["limitations"]]
    (out/"report.md").write_text("\n".join(lines)+"\n", encoding="utf-8")


def plot(out, result):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return "matplotlib unavailable; JSON and CSV remain complete"
    bins = [r for r in result["stratified_bins"] if r["status"] == "descriptive"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    groups = defaultdict(list)
    for r in bins:
        groups[(r["category_id"], r["size"], r["source"])].append(r)
    for key, rows in groups.items():
        rows.sort(key=lambda r: r["bin"])
        for ax, metric in zip(axes, ("oracle_iou_mean", "miss_rate")):
            ax.plot([r["offset_median"] for r in rows], [r[metric] for r in rows], "o-", label=f"{key[0]}/{key[1]}/{key[2]}")
    for ax, name in zip(axes, ("Best-query IoU (geometry ceiling)", "Miss rate at primary score")):
        ax.set(xlabel="Brightness evidence offset / GT diagonal", ylabel=name)
        ax.grid(alpha=.2)
        if groups:
            ax.legend(fontsize=7)
        else:
            ax.text(.5, .5, "Insufficient samples per stratum", ha="center", transform=ax.transAxes)
    fig.tight_layout()
    fig.savefig(out/"stratified_diagnostics.png", dpi=180)
    plt.close(fig)
    return "stratified_diagnostics.png"


def analyze_exports(out, annotations, image_root, p, topk, image_hashes, sample_set):
    inference = out/"inference"
    completed = json.loads((inference/"eval_complete.json").read_text(encoding="utf-8"))
    expected_ids = completed["metrics"]["encoder"]["imgIds"]
    files = list(inference.glob("queries_rank*/image_*.npz"))
    mapping = {}
    for path in files:
        key = int(path.stem.split("_")[-1])
        if key in mapping:
            raise ValueError(f"duplicate query export {key}")
        mapping[key] = path
    if set(mapping) != set(expected_ids):
        raise ValueError("query exports differ from evaluated image IDs")
    image_map = {im["id"]: im for im in annotations["images"]}
    ann_map = defaultdict(list)
    for a in annotations["annotations"]:
        ann_map[a["image_id"]].append(a)
    gt_rows, pred_rows = [], []
    for index, image_id in enumerate(sorted(mapping)):
        image = image_map[image_id]
        path = Path(image_root)/image["file_name"]
        if sha(path) != image_hashes[str(image_id)]:
            raise ValueError(f"image changed since preflight/inference: {path}")
        gray, mode = image_array(path)
        with np.load(mapping[image_id], allow_pickle=False) as data:
            gs, ps = analyze_image(image, ann_map[image_id], data, gray, mode, p, topk)
        gt_rows.extend(gs); pred_rows.extend(ps)
        if index % 50 == 0:
            print(f"Image evidence: {index+1}/{len(mapping)}, GT rows={len(gt_rows)}, prediction rows={len(pred_rows)}", flush=True)
    write_csv(out/"gt_evidence.csv", gt_rows)
    write_csv(out/"prediction_background.csv", pred_rows)
    result = summarize(gt_rows, pred_rows, p)
    result.update({"sample_set": sample_set, "unique_images": len(mapping), "protocol": p,
                   "official_COCO_metrics": completed["metrics"], "plot": plot(out, result)})
    dump(out/"diagnostics.json", result)
    report(out, result, p, sample_set)
    dump(out/"analysis_complete.json", {"sample_set": sample_set, "unique_images": len(mapping),
                                      "gt_rows": len(gt_rows), "prediction_rows": len(pred_rows),
                                      "diagnostics_sha256": sha(out/"diagnostics.json")})
    print(f"Completed {sample_set}: {out/'report.md'}", flush=True)


def validate_annotations(value):
    images = {im["id"]: im for im in value["images"]}
    for a in value["annotations"]:
        im = images[a["image_id"]]
        x, y, w, h = a["bbox"]
        if min(w, h) <= 0 or x < 0 or y < 0 or x+w > im["width"] or y+h > im["height"]:
            raise ValueError(f"invalid/out-of-image GT {a['id']}; review annotation rather than silently clipping")


def run(args, out, p):
    import sar_audit as audit
    audit.runtime()
    torch = audit.torch
    world, rank, local = [int(os.environ.get(k, default)) for k, default in (("WORLD_SIZE", 1), ("RANK", 0), ("LOCAL_RANK", 0))]
    if world > 1:
        if not torch.cuda.is_available():
            raise ValueError("multi-rank inference requires CUDA")
        torch.cuda.set_device(local)
        torch.distributed.init_process_group("nccl")
    try:
        cfg = audit.fresh_config(str(Path(args.config).resolve()))
        original_config = json.loads(json.dumps(cfg.yaml_cfg))
        if cfg.yaml_cfg.get("remap_mscoco_category"):
            raise ValueError("direct category IDs required for this OGSOD diagnostic")
        val = cfg.yaml_cfg["val_dataloader"]["dataset"]
        train = cfg.yaml_cfg["train_dataloader"]["dataset"]
        val_info, annotations = audit.annotation_audit(val["ann_file"], val["img_folder"], cfg.yaml_cfg["num_classes"])
        train_info, train_data = audit.annotation_audit(train["ann_file"], train["img_folder"], cfg.yaml_cfg["num_classes"])
        if train_info["categories"] != val_info["categories"]:
            raise ValueError("train and validation category schemas differ")
        if {im["file_name"] for im in annotations["images"]} & {im["file_name"] for im in train_data["images"]}:
            raise ValueError("train/val image file names overlap")
        validate_annotations(annotations)
        transforms = val["transforms"]["ops"]
        if [op["type"] for op in transforms] != ["Resize", "ConvertPILImage"]:
            raise ValueError("expected deterministic baseline validation Resize + ConvertPILImage; review differing transforms")
        hashes = {str(im["id"]): sha(Path(val["img_folder"])/im["file_name"]) for im in annotations["images"]}
        identity = {"git_commit": audit.git("rev-parse", "HEAD"), "config_path": str(Path(args.config).resolve()),
                    "resolved_config": original_config, "checkpoint": str(Path(args.checkpoint).resolve()),
                    "checkpoint_sha256": sha(args.checkpoint), "weights": args.weights,
                    "train_annotations_sha256": train_info["sha256"], "val_annotations_sha256": val_info["sha256"],
                    "validation_image_sha256": hashes, "protocol": p, "world_size": world, "eval_limit": args.eval_limit,
                    "analysis_source_sha256": sha(__file__), "audit_source_sha256": sha(audit.__file__),
                    "repository_sources": {str(f.relative_to(ROOT)): sha(f) for f in (ROOT/"src").rglob("*.py")},
                    "training_run_commit": args.training_run_commit,
                    "pixel_near_duplicate_check": "not_performed; filename disjointness does not establish source independence"}
        identity_path = out/"identity.json"
        if identity_path.exists():
            if json.loads(identity_path.read_text(encoding="utf-8")) != identity:
                raise ValueError("resume refused: checkpoint/data/source/config/protocol/process count differs")
        elif args.resume_dir:
            raise ValueError("resume directory lacks identity.json")
        elif rank == 0:
            dump(identity_path, identity)
        if world > 1:
            torch.distributed.barrier()
        random.seed(p["seed"]); np.random.seed(p["seed"]); torch.manual_seed(p["seed"])
        device = torch.device("cpu" if args.preflight else f"cuda:{local}" if torch.cuda.is_available() else "cpu")
        # Initialization is fully replaced by strict checkpoint loading; no download.
        cfg.yaml_cfg["PResNet"]["pretrained"] = False
        model = cfg.model.to(device).eval()
        loaded = audit.weight_load(model, args.checkpoint, args.weights)
        if not loaded["strict_load_passed"]:
            if rank == 0:
                dump(out/"weight_load.json", loaded)
            raise ValueError("strict checkpoint load failed; see weight_load.json")
        post = cfg.postprocessor
        if not post.use_focal_loss or post.final_score_method != "default" or any(getattr(post, k) != 0 for k in ("final_quality_gamma", "oracle_final_iou_gamma", "class_aware_oracle_final_iou_gamma", "pairwise_class_aware_oracle_beta")):
            raise ValueError("requires original sigmoid class-pair TopK; no oracle/reranker allowed")
        if rank == 0:
            loaded.update({"torch": torch.__version__, "device": str(device), "world_size": world,
                           "evaluation_precision": "FP32", "training_global_batch_size": original_config["train_dataloader"].get("total_batch_size"),
                           "historical_training_provenance": "provided_commit_only" if args.training_run_commit else "unknown; audit commit is not training commit",
                           "num_classes": original_config["num_classes"], "validation_images": len(annotations["images"])})
            dump(out/"weight_load.json", loaded)
        if args.preflight:
            if rank == 0:
                dump(out/"preflight_complete.json", {"status": "strict_weights_config_annotations_image_hashes_checked_NO_evaluation"})
                print("Preflight passed. No evaluation or detector update performed.", flush=True)
            return
        estimated_bytes = len(annotations["images"]) * model.decoder.num_queries * (model.decoder.num_layers+1) * (original_config["num_classes"]+4) * 12
        if shutil.disk_usage(out).free < estimated_bytes + 1024**3:
            raise OSError("insufficient free disk for query exports/diagnostics and 1 GiB reserve")
        if rank == 0:
            import faster_coco_eval
            dump(out/"runtime.json", {"python": sys.version, "torch": torch.__version__, "numpy": np.__version__,
                 "faster_coco_eval": faster_coco_eval.__version__, "seed": p["seed"], "precision": "FP32",
                 "world_size": world, "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
                 "visible_GPU_names": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
                 "free_disk_bytes": shutil.disk_usage(out).free, "estimated_query_bytes": estimated_bytes,
                 "COCO_categories": annotations["categories"], "validation_transforms": transforms,
                 "no_optimizer_steps": True, "no_parameter_updates": True})
        inference = out/"inference"
        inference.mkdir(exist_ok=True)
        # Enrich the existing tested exporter without changing detector or legacy audit.
        original_geometries = audit.geometries
        def with_exact_pairs(stages, image, annotations, processor, output_path):
            result = original_geometries(stages, image, annotations, processor, output_path)
            with np.load(output_path, allow_pickle=False) as value:
                arrays = {key: value[key] for key in value.files}
            for name, boxes, logits in stages:
                pair_ids = logits.sigmoid().flatten().topk(min(processor.num_top_queries, logits.numel())).indices.cpu().numpy()
                arrays[f"{name}_actual_output_pair_ids"] = pair_ids
                arrays[f"{name}_actual_output_scores"] = result[2][name]["scores"].numpy()
                arrays[f"{name}_actual_output_boxes_xyxy"] = result[2][name]["boxes"].numpy()
                if not np.array_equal(pair_ids % logits.shape[-1], result[2][name]["labels"].numpy()):
                    raise ValueError("query-class pair IDs disagree with postprocessed labels")
            np.savez_compressed(output_path, **arrays)
            return result
        audit.geometries = with_exact_pairs
        try:
            if not (inference/"eval_complete.json").exists():
                audit.run_eval("baseline_evidence", model, cfg, annotations, inference, device, rank, world, args.eval_limit)
        finally:
            audit.geometries = original_geometries
        if world > 1:
            torch.distributed.barrier()
            # Release the process group before rank-0 CPU statistics; no long NCCL wait.
            torch.distributed.destroy_process_group()
        image_root, topk = val["img_folder"], post.num_top_queries
        synthetic = bool(original_config.get("evidence_audit_synthetic_fixture", False))
        del model, cfg, post
        if device.type == "cuda":
            torch.cuda.empty_cache()
        if rank == 0 and not (out/"analysis_complete.json").exists():
            sample_set = "SYNTHETIC_INTEGRATION_NOT_SAR_EVIDENCE" if synthetic else "limited_smoke_NOT_full_validation" if args.eval_limit else "full_validation"
            analyze_exports(out, annotations, image_root, p, topk, hashes, sample_set)
    finally:
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


def self_test(out, p):
    """Generated deterministic counterexamples. Does not validate a SAR hypothesis."""
    rng = np.random.default_rng(0)
    gray = np.full((128, 128), 10.)
    box = np.array([40., 40., 80., 80.])
    gray[57:63, 57:63] = 100
    symmetric = evidence(gray, box, [box], p, .8)
    assert symmetric["proxy_status"] == "valid" and symmetric["evidence_offset"] < 1e-10
    gray[57:63, 57:63] = 10; gray[57:63, 69:75] = 100
    shifted = evidence(gray, box, [box], p, .8)
    assert math.isclose(shifted["evidence_dx"], 12., abs_tol=1e-10)
    assert math.isclose(shifted["evidence_offset"], 12/math.sqrt(3200), abs_tol=1e-10)
    assert evidence(np.ones((128, 128)), box, [box], p, .8)["proxy_status"] == "no_positive_bright_evidence"
    assert evidence(gray, [0, 0, 2, 2], [], p, .8)["proxy_status"] == "insufficient_pixels"
    contaminated = gray.copy(); contaminated[40:80, 10:30] = 10000
    before = evidence(contaminated, box, [box], p, .8)
    after = evidence(contaminated, box, [box, [10, 40, 30, 80]], p, .8)
    assert before["background_tail_z"] > after["background_tail_z"]
    ann = {"id": 1, "image_id": 1, "category_id": 0, "bbox": [40, 40, 40, 40], "area": 1600}
    crowd = {"id": 2, "image_id": 1, "category_id": 0, "bbox": [0, 0, 30, 30], "area": 900, "iscrowd": 1}
    boxes = np.array([box, box, box, [2, 2, 10, 10], [90, 90, 110, 110]])
    _, _, matches, kinds = match_predictions(boxes, np.array([0, 0, 1, 0, 0]), np.array([.9, .8, .7, .6, .5]), [ann, crowd], p)
    assert matches == {0: 0} and kinds == ["TP", "duplicate", "class_error", "ignored_region", "background_candidate"]
    assert math.isclose(geometry(box+np.array([4, 0, 4, 0]), box, shifted)["shift_toward_evidence"], 4/math.sqrt(3200))
    # Controlling perfectly shared size removes a spurious brightness-error correlation.
    rows = [{"image_id": i//2, "category_id": i%2, "source": "unknown", "area": float(np.exp(i/20)),
             "aspect": float(1+rng.random()), "x": float(i), "y": float(i)} for i in range(80)]
    test_p = dict(p, bootstrap_repetitions=30)
    controlled = association(rows, "x", ["y"], test_p)
    assert controlled["outcomes"]["y"]["partial_rank_rho"] is None
    # Direction and repeatability when a relation remains after nuisance control.
    for r in rows:
        r["x"] = float(rng.random()); r["y"] = r["x"] + float(rng.normal(0, .03))
    signal = association(rows, "x", ["y"], test_p)
    assert signal["outcomes"]["y"]["partial_rank_rho"] > .9
    assert signal == association(rows, "x", ["y"], test_p)
    fixture = out/"fixture"; fixture.mkdir()
    image = {"id": 1, "file_name": "1.png", "width": 128, "height": 128}
    Image.fromarray(gray.astype(np.uint8)).save(fixture/"1.png")
    normalized = np.column_stack(((boxes[:, :2]+boxes[:, 2:])/256, (boxes[:, 2:]-boxes[:, :2])/128))
    logits = np.full((5, 2), -10.)
    logits[0, 0], logits[1, 0], logits[2, 1], logits[3, 0], logits[4, 0] = 3, 2, 1.5, 1, .5
    data = {"stage_names": np.array(["encoder", "decoder_0"]), "original_image_wh": np.array([128, 128]),
            "GT_annotation_ids": np.array([1]), "GT_boxes_xyxy": box[None]}
    for stage in data["stage_names"]:
        data[f"{stage}_boxes_cxcywh"] = normalized
        data[f"{stage}_logits"] = logits
        data[f"{stage}_actual_output_pair_ids"] = np.argsort(-logits.ravel(), kind="stable")[:5]
        pairs = data[f"{stage}_actual_output_pair_ids"]
        data[f"{stage}_actual_output_boxes_xyxy"] = boxes[pairs//2]
        data[f"{stage}_actual_output_scores"] = (1/(1+np.exp(-logits))).ravel()[pairs]
    gt, pred = analyze_image(image, [ann, crowd], data, gray, "single_channel", p, 5)
    assert len(gt) == 3 and all(r["failure"] == "detected_precise" for r in gt)
    assert Counter(r["kind"] for r in pred if r["score_threshold"] == p["primary_score"])["background_candidate"] == 1
    empty = dict(data, GT_annotation_ids=np.array([], dtype=int), GT_boxes_xyxy=np.empty((0, 4)))
    empty_gt, empty_pred = analyze_image(image, [], empty, gray, "single_channel", p, 5)
    assert not empty_gt and all(r["kind"] == "background_candidate" for r in empty_pred)
    write_csv(out/"synthetic_gt_evidence.csv", gt)
    write_csv(out/"synthetic_prediction_background.csv", pred)
    result = summarize(gt, pred, test_p)
    dump(out/"diagnostics.json", result)
    report(out, result, test_p, "SYNTHETIC_SELF_TEST_NOT_SAR_EVIDENCE")
    plot(out, result)
    dump(out/"self_test.json", {"status": "passed_SYNTHETIC_ONLY", "checks": ["pixel-centre units", "symmetric and displaced bright evidence",
        "flat/dark/tiny image invalidity", "GT exclusion from background ring", "unique matching, duplicates, class error, crowd IOA",
        "signed geometry shift", "size confounding control", "image-cluster bootstrap repeatability", "query pairs and end-to-end CSV/report"],
        "controlled_signal": signal})
    print("Synthetic self-test passed. No real SAR hypothesis was evaluated.", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", default=str(ROOT/"configs/analysis/sar_evidence_v1.yml"))
    parser.add_argument("--config")
    parser.add_argument("--checkpoint")
    parser.add_argument("--weights", choices=("ema", "model"), default="ema")
    parser.add_argument("--training-run-commit", help="Historical training commit, if known; never inferred from current HEAD")
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--eval-limit", type=int, help="Smoke test only, never a full-validation conclusion")
    dirs = parser.add_mutually_exclusive_group(required=True)
    dirs.add_argument("--output")
    dirs.add_argument("--resume-dir")
    args = parser.parse_args()
    dependencies()
    p = protocol(args.protocol)
    if args.eval_limit is not None and args.eval_limit < 1:
        parser.error("eval-limit must be positive")
    if not args.self_test and (not args.config or not args.checkpoint):
        parser.error("real run requires --config and explicit --checkpoint")
    if args.self_test and (args.preflight or args.resume_dir or args.config or args.checkpoint):
        parser.error("self-test cannot be combined with real-run options or resume")
    out = Path(args.resume_dir or args.output).resolve()
    rank = int(os.environ.get("RANK", 0))
    if args.resume_dir:
        if not out.is_dir():
            parser.error("resume directory does not exist")
    else:
        # torchrun ranks coordinate directory creation before any artifacts are written.
        # Launcher creates only logs/PID; every other existing file is refused.
        if rank == 0 and out.exists() and any(f.name not in ("console.log", "launcher.pid") for f in out.iterdir()):
            parser.error("fresh output contains existing artifacts; use new output or matching --resume-dir")
        out.mkdir(parents=True, exist_ok=True)
    if args.self_test:
        self_test(out, p)
    else:
        run(args, out, p)


if __name__ == "__main__":
    main()
