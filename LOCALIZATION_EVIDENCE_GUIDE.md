# SAR 定位瓶颈：P1 离线证据与 P2 固定 pilot

## 1. 本次改动与状态

新增 `tools/localization_diagnosis.py`（P1，无推理/训练）、`tools/prepare_dfine_pilot.py`（生成独立官方 D-FINE 配置）、`tools/localization_pilot_check.py`（合成检查或两模型同协议评估）、`tools/test_localization_diagnosis.py`、两份 `configs/diagnosis/` 配置以及本指南和 `D_FINE_FEASIBILITY.md`。不改 `src/`、E0/E1/E2/E3/E4 配置或训练设置。

已知服务器结果目录：`/data2/zxy/sar/experiments/ogsod_rtdetrv2_r18_baseline_error_analysis_best_recovery_mirror`。文件、原图与 checkpoint 不在本地；**本地没有实际 P1 指标、SAR 人工图像复核或 D-FINE 性能结果**。

P1 自动检查 checkpoint SHA/epoch/EMA/strict-load记录、完整配置、原始val GT、预测字节hash、已有导出shards（若仍存在）、train/val文件名交集，独立复算 COCO 并核对旧 `coco_metrics.json`/`error_summary.csv`。记录旧报告/CSV当前hash；这不证明历史报告从未被人为改动。旧 exporter 未记录推理Git commit，不能用CPU recovery commit代替：留待检查当时console/git记录。

P1不使用旧80e之外的预测、不需要TIDE、不重新跑模型。完整漏检不分配虚构几何误差；仅COCO一对一TP50框用于中心/尺度/宽高比数学分析。中心、面积尺度、宽高比 oracle IoU增益独立、不可相加；边界偏移不是第四个独立自由度。不能从坐标误差推断模糊、散射偏移或标签噪声。

## 2. 本地验证

```powershell
conda run -n pytorch python -X utf8=0 tools/test_localization_diagnosis.py
conda run -n pytorch python -X utf8=0 tools/test_baseline_error_analysis.py
conda run -n pytorch python -X utf8=0 -m py_compile tools/localization_diagnosis.py tools/prepare_dfine_pilot.py tools/localization_pilot_check.py tools/test_localization_diagnosis.py
```

实际检查记录见 `D_FINE_FEASIBILITY.md`；所有模型检查都是合成输入、无参数更新。不能用此记录声称服务器数据或训练已通过。

## 3. Git 传输

分支：`codex/ogsod-localization-evidence`。检查服务器工作树，若存在修改或分叉先处理，不能强制覆盖：

```bash
cd /home/zxy/sar/repos/rtdetrv2_pytorch
git status --short
git fetch origin
git switch codex/ogsod-localization-evidence
git pull --ff-only origin codex/ogsod-localization-evidence
conda activate rtdetr_zxy
```

## 4. 现在只执行 P1（CPU、复用预测）

下面使用上一轮原始GPU导出目录和你确认的CPU恢复分析目录。两者必须保留 `export_metadata.json`/`predictions.json` 等实际证据；若原始导出目录已删除，先找回它，不能直接将恢复目录的 metadata 当作推理来源。

```bash
cd /home/zxy/sar/repos/rtdetrv2_pytorch
conda activate rtdetr_zxy
SRC=/data2/zxy/sar/experiments/ogsod_rtdetrv2_r18_baseline_error_analysis_best
ANALYSIS=/data2/zxy/sar/experiments/ogsod_rtdetrv2_r18_baseline_error_analysis_best_recovery_mirror
OUT=/data2/zxy/sar/experiments/ogsod_baseline_localization_diagnosis_best
test -r "$SRC/export_metadata.json"
test -r "$ANALYSIS/COMPLETE.json"
df -h /data2/zxy/sar/experiments
python -c 'import numpy, matplotlib, PIL, pycocotools, torch, yaml'
if [ -e "$OUT" ]; then
  echo "输出目录已存在，请检查后选择新的目录；本次未启动。"
else
  mkdir -p "$OUT"
  nohup python tools/localization_diagnosis.py \
    --export-dir "$SRC" \
    --analysis-dir "$ANALYSIS" \
    --checkpoint /data2/zxy/sar/experiments/ogsod_rtdetrv2_r18_80e_baseline_gbs64_gpu2_zxy/best.pth \
    --config configs/rtdetrv2/rtdetrv2_r18vd_80e_ogsod_baseline_gbs64_gpu2_zxy.yml \
    --train-annotations /home/zxy/sar/datasets/OGSOD-1.0/sar/RTDETR_COCO/train.json \
    --images /home/zxy/sar/datasets/OGSOD-1.0/sar \
    --output "$OUT" > "$OUT/console.log" 2>&1 &
  ANALYSIS_PID=$!
  echo "$ANALYSIS_PID" | tee "$OUT/launcher.pid"
  disown
fi
```

工具会核对这些路径与原export完整配置，不悄悄改数据路径。若路径不一致，先查原 `export_metadata.json`，定位真实数据路径/配置差异。P1无需GPU，不运行torchrun。

输出：

- `LOCALIZATION_DIAGNOSIS.md`：每类别/尺寸AP75、AR75、FN75比例、oracle误差、五个选题问题与不确定性。
- `provenance.json`：hash、split、配置、epoch、分析commit、推理commit缺失项与旧COCO复算核对。
- `gt_geometry.csv`、`matched_box_errors.csv`、`class_size_statistics.csv`：原图/输入尺寸、长宽比、中心/宽高/四边误差、IoU、单因素oracle。
- `bridge_size_iou.png/svg`、`harbor_size_iou.png/svg`、`storage_tank_size_iou.png/svg`。
- `examples/`：最多20例典型TP50但IoU<.75框的原图与保持比例放大图；原FN75图通过 `example_index.csv` 链接，保留源图。
- `manual_review.csv`：所有项初始化pending/unknown；结合原图与GT填写证据，不能把自动画框当作人工审阅。
- `COMPLETE.json`：`numerical_analysis_complete` 与 `manual_review: pending` 分开。

短边分组 `<8`、`[8,16)`、`[16,32)`、`>=32` 是原val变换后的真实尺寸；不是COCO small/medium/large。保留其它尺寸GT作为ignore，分组外未匹配预测也ignore，防止污染分组AP。分组AP/AR重新匹配；FN75条件比例用全验证集匹配，因此二者不必互补。还记录P3短边cell数和1px偏移的纯几何敏感性：只能支持高分辨率研究假设，不能证明网络stride或SAR机制是原因。

## 5. P2 只准备和检查，不自动训练

固定官方D-FINE源码，独立环境保护原环境。以下目录应为新目录；已存在时先查commit/修改，不能覆盖或reset：

```bash
cd /home/zxy/sar/repos
git clone https://github.com/Peterande/D-FINE.git dfine_ogsod_pilot
cd /home/zxy/sar/repos/dfine_ogsod_pilot
git checkout --detach 956d1709314c2c6a4df6f34de232054578a7449f
git status --short
conda create -n dfine_ogsod --clone rtdetr_zxy -y
conda activate dfine_ogsod
python -c 'import importlib.metadata as m; print("\n".join(n+"=="+m.version(n) for n in ("torch","torchvision","numpy","scipy","faster-coco-eval","pycocotools","Pillow")))' > /tmp/ogsod_dfine_core_constraints.txt
python -m pip install -c /tmp/ogsod_dfine_core_constraints.txt calflops==0.3.2 loguru==0.7.3 accelerate==1.10.1 transformers==4.57.6
python -c 'import calflops, loguru, accelerate, transformers, pycocotools'
cd /home/zxy/sar/repos/rtdetrv2_pytorch
python tools/prepare_dfine_pilot.py --dfine-repo /home/zxy/sar/repos/dfine_ogsod_pilot
```

如网络超时可加 `--index-url https://pypi.tuna.tsinghua.edu.cn/simple --timeout 120`；保留constraints，不升级原环境依赖。若当前Python版本不支持依赖，保留错误并处理，不静默换核心库。

生成配置 `/home/zxy/sar/repos/dfine_ogsod_pilot/configs/ogsod/dfine_r18_ogsod_pilot20.yml` 和 `.protocol.json`，保存共享协议、split SHA、upstream commit；没有启动训练。准备脚本拒绝重复覆盖配置。

服务器用合成数据在CPU上检查两模型（不占用训练GPU）：

```bash
cd /home/zxy/sar/repos/rtdetrv2_pytorch
conda activate rtdetr_zxy
python tools/localization_pilot_check.py --repo /home/zxy/sar/repos/rtdetrv2_pytorch \
  --config configs/diagnosis/rtdetrv2_r18_ogsod_pilot20.yml \
  --output /data2/zxy/sar/experiments/ogsod_r18_pilot20_preflight \
  --save-backbone /tmp/ogsod_shared_r18_backbone.pth
conda activate dfine_ogsod
python tools/localization_pilot_check.py --repo /home/zxy/sar/repos/dfine_ogsod_pilot \
  --config /home/zxy/sar/repos/dfine_ogsod_pilot/configs/ogsod/dfine_r18_ogsod_pilot20.yml \
  --output /data2/zxy/sar/experiments/ogsod_dfine_r18_pilot20_preflight \
  --backbone-state /tmp/ogsod_shared_r18_backbone.pth
```

合成检查不验证真实数据AMP/DDP训练；查看 `validation.json`。当前PResNet使用同一个下载URL，但缓存目录不同。为保证预训练字节相同，从同一缓存文件复制至官方D-FINE的 `weight/`，若已有不同文件则拒绝覆盖：

```bash
python - <<'PY'
import hashlib, pathlib, shutil, torch
url = 'https://github.com/lyuwenyu/storage/releases/download/v0.1/ResNet18_vd_pretrained_from_paddle.pth'
torch.hub.load_state_dict_from_url(url, map_location='cpu')
source = pathlib.Path(torch.hub.get_dir())/'checkpoints'/url.rsplit('/',1)[1]
target = pathlib.Path('/home/zxy/sar/repos/dfine_ogsod_pilot/weight')/source.name
target.parent.mkdir(parents=True, exist_ok=True)
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
if target.exists() and sha(target) != sha(source):
    raise RuntimeError('Existing D-FINE pretrained bytes differ; inspect first')
if not target.exists():
    shutil.copy2(source, target)
print('Shared ImageNet SHA256:', sha(source))
PY
```

保存该SHA到两边实验记录；正式初始化仍是各自 `pretrained: True`，不使用这里的合成backbone state。

## 6. 推荐的唯一20e对照命令（本次未执行）

先完成P1与服务器preflight。两次运行顺序执行，避免共享GPU抢占；确认GPU空闲、磁盘空间、全局batch64、data split和预训练文件。不自动跑80/120e。

E0 pilot：

```bash
cd /home/zxy/sar/repos/rtdetrv2_pytorch
conda activate rtdetr_zxy
OUT=/data2/zxy/sar/experiments/ogsod_rtdetrv2_r18_pilot20_seed0
df -h /data2/zxy/sar/experiments
nvidia-smi
if [ -e "$OUT" ]; then
  echo "目录存在，先检查；不启动新训练。"
else
  mkdir -p "$OUT"
  git rev-parse HEAD > "$OUT/source_commit.txt"
  nohup env CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 tools/train.py \
    -c configs/diagnosis/rtdetrv2_r18_ogsod_pilot20.yml --use-amp --seed 0 --output-dir "$OUT" \
    > "$OUT/console.log" 2>&1 &
  TRAIN_PID=$!
  echo "$TRAIN_PID" | tee "$OUT/launcher.pid"
  disown
fi
```

E0结束并释放GPU后，D-FINE pilot：

```bash
cd /home/zxy/sar/repos/dfine_ogsod_pilot
conda activate dfine_ogsod
OUT=/data2/zxy/sar/experiments/ogsod_dfine_r18_pilot20_seed0
df -h /data2/zxy/sar/experiments
nvidia-smi
if [ -e "$OUT" ]; then
  echo "目录存在，先检查；不启动新训练。"
else
  mkdir -p "$OUT"
  git rev-parse HEAD > "$OUT/source_commit.txt"
  cp configs/ogsod/dfine_r18_ogsod_pilot20.protocol.json "$OUT/protocol.json"
  nohup env CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 train.py \
    -c configs/ogsod/dfine_r18_ogsod_pilot20.yml --use-amp --seed 0 --output-dir "$OUT" \
    > "$OUT/console.log" 2>&1 &
  TRAIN_PID=$!
  echo "$TRAIN_PID" | tee "$OUT/launcher.pid"
  disown
fi
```

两边保留原loss。配置只截断同一80e策略的前20e，不缩warmup、不做sweep；不能将任一20e结果与旧80e best比较。

## 7. best checkpoint 的共同评估

顺序评估，GPU空闲；相同Torch/COCO环境、FP32、batch1固定640测速（不含数据加载），20次warmup/100次测量。原生evaluation输出继续保留，本工具补充同协议JSON与分类别AP75。

```bash
cd /home/zxy/sar/repos/rtdetrv2_pytorch
conda activate rtdetr_zxy
env CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 tools/localization_pilot_check.py \
  --repo /home/zxy/sar/repos/rtdetrv2_pytorch \
  --config configs/diagnosis/rtdetrv2_r18_ogsod_pilot20.yml \
  --checkpoint /data2/zxy/sar/experiments/ogsod_rtdetrv2_r18_pilot20_seed0/best.pth \
  --output /data2/zxy/sar/experiments/ogsod_rtdetrv2_r18_pilot20_seed0_eval_best --benchmark
conda activate dfine_ogsod
env CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 tools/localization_pilot_check.py \
  --repo /home/zxy/sar/repos/dfine_ogsod_pilot \
  --config /home/zxy/sar/repos/dfine_ogsod_pilot/configs/ogsod/dfine_r18_ogsod_pilot20.yml \
  --checkpoint /data2/zxy/sar/experiments/ogsod_dfine_r18_pilot20_seed0/best_stg1.pth \
  --output /data2/zxy/sar/experiments/ogsod_dfine_r18_pilot20_seed0_eval_best --benchmark
```

D-FINE20e停在stage1，因此最佳文件为 `best_stg1.pth`。共同评估严格加载EMA，不需要下载预训练权重。重复评估 last 时将 checkpoint 换为各自 `last.pth`，分别使用新输出 `_eval_last`；其余设置相同。

每个 `_eval_best/coco_metrics.json`：AP、AP50、AP75、APs、AR75、`class_AP75.Bridge` / `class_AP75.Storage Tank` 等（0–1小数）。`evaluation_manifest.json`：参数、checkpoint/config/val/prediction SHA、commit、latency；训练时间使用每个console末尾 `Training time`。检查两边 `log.txt` 确实完成epoch0–19，不将未跑完的一方当20e结果。

## 8. 监控、输出与恢复

P1检查：

```bash
OUT=/data2/zxy/sar/experiments/ogsod_baseline_localization_diagnosis_best
tail -n 100 -f "$OUT/console.log"
# Ctrl+C只结束tail，再执行下面命令。
ps -fp "$(cat "$OUT/launcher.pid")"
python -m json.tool "$OUT/COMPLETE.json"
cat "$OUT/LOCALIZATION_DIAGNOSIS.md"
```

P1不写checkpoint。若失败，修正来源/依赖后使用新目录（例如 `_retry1`），不覆盖既有证据；无需恢复训练或重新推理。

pilot监控时选对应OUT（E0或D-FINE），分别执行：

```bash
OUT=/data2/zxy/sar/experiments/ogsod_rtdetrv2_r18_pilot20_seed0
tail -n 100 -f "$OUT/console.log"
ps -fp "$(cat "$OUT/launcher.pid")"
watch -n 2 nvidia-smi
tail -n 1 "$OUT/log.txt" | python -m json.tool
rg 'Training time' "$OUT/console.log"
tensorboard --logdir "$OUT/summary" --host 0.0.0.0 --port 6006
```

`tail -f`、`watch`、TensorBoard会阻塞当前终端，按Ctrl+C退出或另开终端，不是将这些命令排队后继续执行。D-FINE OUT为 `/data2/zxy/sar/experiments/ogsod_dfine_r18_pilot20_seed0`。last/periodic/eval/summary均在各自OUT。

仅中断的pilot需要resume，先确认没有旧进程、last可读取、磁盘空间充足；保持同配置、GPU、seed、AMP，不用tuning：

```bash
cd /home/zxy/sar/repos/rtdetrv2_pytorch
conda activate rtdetr_zxy
OUT=/data2/zxy/sar/experiments/ogsod_rtdetrv2_r18_pilot20_seed0
nohup env CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 tools/train.py \
  -c configs/diagnosis/rtdetrv2_r18_ogsod_pilot20.yml --resume "$OUT/last.pth" \
  --use-amp --seed 0 --output-dir "$OUT" > "$OUT/resume.log" 2>&1 &
TRAIN_PID=$!
echo "$TRAIN_PID" | tee "$OUT/launcher.pid"
disown
```

```bash
cd /home/zxy/sar/repos/dfine_ogsod_pilot
conda activate dfine_ogsod
OUT=/data2/zxy/sar/experiments/ogsod_dfine_r18_pilot20_seed0
nohup env CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 train.py \
  -c configs/ogsod/dfine_r18_ogsod_pilot20.yml --resume "$OUT/last.pth" \
  --use-amp --seed 0 --output-dir "$OUT" > "$OUT/resume.log" 2>&1 &
TRAIN_PID=$!
echo "$TRAIN_PID" | tee "$OUT/launcher.pid"
disown
```

## 9. 五个问题的证据要求

目前实际答案均待服务器数据与图像复核：主要中心/尺度/长宽比误差看逐类oracle gain/恢复75率及有符号边界偏移；极小集中性看条件FN75比例、AP75/AR75和样本量；高分辨率仅能形成假设，不能因小目标低IoU就判定有效；D-FINE优势看同预算新pilot；论文切入点还要排除标注问题。`manual_review.csv` 未填写证据前，不报告散射偏移、模糊边界或标签质量作为确定原因。

执行P1后提供 `LOCALIZATION_DIAGNOSIS.md`、`provenance.json`、分类统计、逐框误差和原图/GT/典型图，才能在本地完成数值判断与人工复核。只返回文件路径不足以读取服务器内容。先提交和运行这些证据工具，不增加网络模块、不改baseline、不自动扩展实验。
