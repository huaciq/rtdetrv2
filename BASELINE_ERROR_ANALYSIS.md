# 一次 baseline 选题前错误分析

本次只增加独立分析工具和合成正确性检查。没有修改 `src/`、模型、训练配置、matcher、loss 或 query selection，也没有运行训练。

checkpoint 固定为：

```text
/data2/zxy/sar/experiments/ogsod_rtdetrv2_r18_80e_baseline_gbs64_gpu2_zxy/best.pth
```

配置固定为 `configs/rtdetrv2/rtdetrv2_r18vd_80e_ogsod_baseline_gbs64_gpu2_zxy.yml`。使用配置指定的 EMA 权重，缺少 EMA 或任何 state_dict 不匹配即失败；不会以非严格加载掩盖问题。完整权重加载后无需再下载 ImageNet 预训练文件。该初始化开关不影响已加载 checkpoint 的推理行为。

## 服务器更新与依赖

本地不能访问服务器 `/data2`，真实分析必须在服务器执行。不要把本地合成测试当作 OGSOD 性能证据。

```bash
cd /home/zxy/sar/repos/rtdetrv2_pytorch
git status --short
git fetch origin
git switch codex/ogsod-baseline-error-analysis
git pull --ff-only origin codex/ogsod-baseline-error-analysis
conda activate rtdetr_zxy
python -c 'import importlib.metadata as m; print("\n".join(n+"=="+m.version(n) for n in ("torch", "torchvision", "numpy", "scipy", "faster-coco-eval", "pycocotools", "Pillow")))' > /tmp/ogsod_error_analysis_constraints.txt
python -m pip install -c /tmp/ogsod_error_analysis_constraints.txt tidecv==1.0.1
python -c 'import torch, faster_coco_eval, pycocotools, tidecv; print(torch.__version__, faster_coco_eval.__version__)'
```

若服务器 worktree 有未提交修改，先检查；不要强制切换或 reset。TIDE 是离线分析依赖，不是模型模块；安装时约束现有核心依赖版本，避免改变 baseline 的数值环境。

启动前工具会检查安装元数据和实际导入，缺少 `tidecv` 或其依赖时在 GPU 推理和结果目录创建之前失败。必须使用 `rtdetr_zxy` 环境里的 `python -m pip`，不能只在 `base` 环境安装。

## 固定流程启动（两张 GPU，只有一次推理）

baseline 配置目前记录的验证路径是：

```text
/home/zxy/sar/datasets/OGSOD-1.0/sar/RTDETR_COCO/val.json
/home/zxy/sar/datasets/OGSOD-1.0/sar
```

这些是已有配置的记录，并非已重新确认的服务器路径。工具会检查 JSON 及每张验证图像是否存在；迁移后的路径需要用 `--annotations`、`--images` 明确指定真实位置，不会猜测 `/data2` 数据目录。

```bash
cd /home/zxy/sar/repos/rtdetrv2_pytorch
conda activate rtdetr_zxy
OUT=/data2/zxy/sar/experiments/ogsod_rtdetrv2_r18_baseline_error_analysis_best
CKPT=/data2/zxy/sar/experiments/ogsod_rtdetrv2_r18_80e_baseline_gbs64_gpu2_zxy/best.pth
CFG=configs/rtdetrv2/rtdetrv2_r18vd_80e_ogsod_baseline_gbs64_gpu2_zxy.yml
test -f "$CKPT" || { echo 'checkpoint missing'; exit 1; }
test -f /home/zxy/sar/datasets/OGSOD-1.0/sar/RTDETR_COCO/val.json || { echo '请先确认迁移后的验证 JSON 路径'; exit 1; }
test -d /home/zxy/sar/datasets/OGSOD-1.0/sar || { echo '请先确认迁移后的图像目录'; exit 1; }
test ! -e "$OUT" || { echo '分析目录已经存在，不覆盖；请先检查已有结果'; exit 1; }
mkdir -p "$OUT"
df -h "$OUT"
nohup env CUDA_VISIBLE_DEVICES=0,1 \
  torchrun --standalone --nproc_per_node=2 tools/baseline_error_analysis.py \
  --config "$CFG" --checkpoint "$CKPT" \
  --output "$OUT" > "$OUT/console.log" 2>&1 &
ANALYSIS_PID=$!
echo "$ANALYSIS_PID" | tee "$OUT/launcher.pid"
disown
```

没有 `--use-amp`：沿用原 `det_engine.evaluate()` 的 FP32 推理。保留原 validation transforms、原 postprocessor 和所有最终预测，不做 score 过滤或额外 NMS。验证集按索引分片，不补齐，不重复图像；合并后检查 image_id 与原验证集完全一致。

推理结束后其他 rank 退出，rank0 继续 CPU 分析。此时 GPU 利用率变为零是正常现象。没有 optimizer、反向传播或 Query Diagnosis hooks。

## 指标口径

- 原 COCO AP50:95、AP50、AP75、APs 及完整12项指标使用仓库相同的 faster-coco-eval。AR75 从 IoU=.75、area=all、maxDets=100 的有效类别 recall 取均值。
- 官方 [TIDE](https://github.com/dbolya/tide) 固定 IoU foreground=.5、background=.1。输出 Cls、Loc、Both、Dupe、Bkg、Miss 的数量和独立 ΔAP@.50（百分点）。独立贡献不能相加。官方 TIDE 是 top100/图/跨类别；COCO 是 top100/图/类别，报告明确保留差异。
- 高置信度固定为 score≥.5，仅用于 FP 明细和示例，不影响 AP、TIDE 或完整预测导出。使用同一份预测、标准 COCO 一对一匹配在 .5/.75 下形成漏检统计，含 crowd 忽略；无需第二次模型推理。
- 漏检、目标定位分组使用 GT 原始面积；背景 FP 无对应 GT，使用预测面积。计数采用互斥 small<32²、medium∈[32²,96²)、large≥96²；COCO AP 保留官方面积区间。
- 每个 GT 同类/任意类 best-IoU 用来区分最终输出缺少重叠、定位不足与类别混淆。它是覆盖上限，允许一个框覆盖多个 GT，不是 COCO recall，也不是 encoder query recall。
- TP/FP 示例定义在 IoU=.5、score≥.5；FN 示例定义在 IoU=.75，包括“已达到 .5、未达到 .75”的目标。各20例，类别/尺度/错误类型轮转，单组每个图像至多一例。若不足20不复制；每例含全图和放大裁剪。

## 输出、监控与恢复

```bash
OUT=/data2/zxy/sar/experiments/ogsod_rtdetrv2_r18_baseline_error_analysis_best
tail -n 80 -f "$OUT/console.log"
ps -fp "$(cat "$OUT/launcher.pid")"
watch -n 2 nvidia-smi
test -f "$OUT/COMPLETE.json" && python -m json.tool "$OUT/COMPLETE.json"
cat "$OUT/ERROR_ANALYSIS.md"
python -m json.tool "$OUT/coco_metrics.json"
python -m json.tool "$OUT/tide_summary.json"
```

不创建 TensorBoard events、训练日志或新 checkpoint。`COMPLETE.json` 是全部步骤完成标记；仅有预测 JSON 不代表分析完成。

| 文件 | 用途 |
|---|---|
| `predictions.json` | 未按分数过滤的完整 COCO 格式最终预测 |
| `validation_gt.json` | 原始验证 GT 的副本 |
| `export_metadata.json` | strict load、EMA/model、checkpoint SHA256、epoch、配置、GPU数 |
| `analysis_manifest.json` | GT/预测 SHA256、Git commit、时间、路径、依赖版本 |
| `coco_metrics.json` | 原 COCO 指标及 AR75 |
| `tide_summary.json` / `tide_errors.csv` | 六类独立贡献与逐条错误 |
| `error_summary.csv` | 总体、类别、尺度的漏检/高置信错误/覆盖分布 |
| `high_confidence_detections.csv` | 逐检测 score、IoU、GT类别、TP/FP及错误标签 |
| `gt_localization.csv` | 逐 GT 的 .5/.75 漏检和高/全置信覆盖证据 |
| `visualizations/{TP,FP,FN}/` | 每组20例全图及裁剪图 |
| `visualization_index.csv` | 图像与类别、检测/GT记录的映射 |
| `ERROR_ANALYSIS.md` | 数值证据、方向优先级及结论边界 |

若 CPU 分析中断但完整 `predictions.json` 已导出，只分析现有预测，不再跑模型，使用新的结果目录：

若日志报 `PackageNotFoundError: tidecv`，先执行上面的依赖安装和导入检查。原调用会在版本记录失败前保存 `predictions.json`、`export_metadata.json` 和 `validation_gt.json`；保留整个原目录。`tail -f` 和 `watch` 都会持续占用终端，按 Ctrl+C 退出监控再输入恢复命令，不要把这些持续监控命令与恢复命令整段连续粘贴。

```bash
cd /home/zxy/sar/repos/rtdetrv2_pytorch
conda activate rtdetr_zxy
OUT=/data2/zxy/sar/experiments/ogsod_rtdetrv2_r18_baseline_error_analysis_best
RECOVERY=/data2/zxy/sar/experiments/ogsod_rtdetrv2_r18_baseline_error_analysis_best_recovery
test -s "$OUT/predictions.json" || { echo '完整预测文件不存在，请先检查原输出'; exit 1; }
test -s "$OUT/validation_gt.json" || { echo '验证 GT 副本不存在，请先检查原输出'; exit 1; }
test ! -e "$RECOVERY" || { echo '恢复结果目录已存在，请先检查'; exit 1; }
mkdir -p "$RECOVERY"
nohup python tools/baseline_error_analysis.py \
  --predictions "$OUT/predictions.json" \
  --annotations "$OUT/validation_gt.json" \
  --images /home/zxy/sar/datasets/OGSOD-1.0/sar \
  --output "$RECOVERY" > "$RECOVERY/console.log" 2>&1 &
ANALYSIS_PID=$!
echo "$ANALYSIS_PID" | tee "$RECOVERY/launcher.pid"
disown
```

上面的 image root 同样需要使用实际验证路径。离线模式把来源标记为外部预测；通过比较旧、新 manifest 的 GT/预测 SHA256 验证对应关系，原 `export_metadata.json` 保留 checkpoint 证据。预测导出尚未完成时不能把零散 rank shards 当成完整结果。

## 本地验证与结论限制

```powershell
conda run -n pytorch python -X utf8=0 -m py_compile tools/baseline_error_analysis.py tools/test_baseline_error_analysis.py
conda run -n pytorch python -X utf8=0 tools/test_baseline_error_analysis.py
```

检查依赖缺失提前失败、严格 EMA 加载、两张合成图的原 baseline 300输出/图、六类错误、COCO one-to-one、crowd、.5/.75区分、空预测与未定义的 oracle、每组20例及裁剪。合成数据和临时 checkpoint 全部位于系统临时目录，不进入 Git。

自动报告按独立 oracle 的最大项及 small 覆盖给出保守方向建议。TIDE Miss 不等于 COCO 全部 FN，漏检不自动证明需要知识迁移；背景 FP 也可能对应未标注目标。典型图需要人工核对后，才可把“与 GT 不重叠”解释为真实 SAR 杂波混淆。单 checkpoint 不能提供 teacher、监督方式或因果机制的对照证据。
