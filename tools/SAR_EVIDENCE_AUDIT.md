# SAR 证据偏移与背景误检：第一轮验证

## 1. What changed

新增 `tools/sar_evidence_audit.py`、`configs/analysis/sar_evidence_v1.yml` 和 `tools/run_sar_evidence_audit.sh`。复用 `tools/sar_audit.py` 的只读 hooks、严格 checkpoint 加载及 COCO 评估。没有修改检测器、损失、训练配置或原有审计脚本。

本次基线配置：`configs/rtdetrv2/rtdetrv2_r18vd_80e_ogsod_baseline_gbs64_gpu2_zxy.yml`；用户指定的权重按项目根目录解析为：

```text
/home/zxy/sar/experiments/ogsod_rtdetrv2_r18_80e_baseline_gbs64_gpu2_zxy/best.pth
```

显式加载 `ema.module`，缺失或结构不匹配会停止；不会退回 model 或随机初始化。推理使用 FP32，不开 AMP；不更新模型，不需要重新训练。保留官方分类分数及 TopK query/class 配对，并存储每层 encoder/decoder 查询。验证集分片不补齐，避免 DDP sampler 重复末尾图像。两个推理进程各自处理不同图像，无梯度同步。

### 验证什么

- **证据偏移与定位**：用原始显示图像中 GT 框内相对背景较亮的像素，计算加权质心。`evidence_offset = ||c_E-c_B|| / sqrt(w²+h²)`。检查偏移与漏检、最佳查询 IoU、中心误差、框面积收缩及朝证据方向偏移的关联。
- **背景与误检**：使用 bbox 外的背景环统计亮尾、MAD 和前景对比。比较 TP 和低重叠的背景误检候选；重复预测、类别错误、部分重叠和 crowd/ignore 区域单独计数。
- **失效位置**：区分 encoder 无几何覆盖、final query 无几何覆盖、类别/分数/输出筛选、唯一匹配竞争，以及检测成功但定位较粗。

亮度质心是**可观测证据代理**，不能写成由 PNG 恢复的真实散射中心。背景统计也不能直接叫校准后的 SCR、ENL、雷达相位或极化特征。

### 预先固定的分析协议

参数集中在 `configs/analysis/sar_evidence_v1.yml`：主证据分位数 0.8，敏感性检查 0.7/0.9；主分数阈值 0.3，备用 0.05/0.5；匹配 IoU 0.5，精定位 IoU 0.75，背景候选最大 IoU < 0.1。诊断匹配是类别一致、分数优先、一对一匹配，**不替代官方 COCO AP**。

背景环：outer scale 2.0，guard scale 1.2，排除所有已知标注框（包括 crowd/ignore）；这些 GT 信息仅用于离线诊断。最少背景像素 64、前景像素 4、正证据像素 3。不够或没有正亮度证据时标记无效，不把偏移记成零。亮像素含分位数阈值上的并列值，权重为 `max(pixel-background_median, 0)`。MAD 接近零的尺度下限由背景自身计算，不由前景峰值决定。

控制类别、log 面积秩、log 长宽比秩，并加入类别与尺寸/长宽比的交互；有明确来源 metadata 时才控制来源。以整幅图像为单位做 500 次 bootstrap，每次重新拟合控制变量。报告偏相关和描述性 95% CI，不给因果结论。低于 30 个对象或 20 幅图像时不估计关联。类别/COCO尺寸/来源内另列三分位组；稀疏或并列分位数组显式标记。

源码、解析配置、checkpoint、标注和每张验证图像都记录 SHA256；resume 比较来源、数据、权重、协议、Git commit、GPU 进程数及 smoke 范围。未提供训练时 commit 时明确记录未知，当前审计 commit 不冒充训练 commit。没有真实来源标签时 `source=unknown`，不从文件名猜传感器。

## 2. Local verification

本地使用 `pytorch` 环境。其自动 site 初始化存在已有 `.pth` 问题，因此仅在忽略的 `outputs/evidence_bootstrap.py` 中追加现有依赖路径，以 `-S` 执行；未改 conda 环境。生成数据和随机 checkpoint 全在 `outputs/`，不提交 Git。

```powershell
& 'D:\Anc\anaconda3\envs\pytorch\python.exe' -X utf8 -S -m py_compile tools/sar_evidence_audit.py
& 'D:\Git\Git\bin\bash.exe' -n tools/run_sar_evidence_audit.sh
& 'D:\Anc\anaconda3\envs\pytorch\python.exe' -X utf8 -S outputs/evidence_bootstrap.py --self-test --output outputs/sar_evidence_20261008_selftest_04
& 'D:\Anc\anaconda3\envs\pytorch\python.exe' -X utf8 -S outputs/evidence_bootstrap.py --config outputs/sar_evidence_20261008_model_fixture/config.yml --checkpoint outputs/sar_evidence_20261008_model_fixture/synthetic_checkpoint.pth --preflight --output outputs/sar_evidence_20261008_preflight_02
& 'D:\Anc\anaconda3\envs\pytorch\python.exe' -X utf8 -S outputs/evidence_bootstrap.py --config outputs/sar_evidence_20261008_model_fixture/config.yml --checkpoint outputs/sar_evidence_20261008_model_fixture/synthetic_checkpoint.pth --output outputs/sar_evidence_20261008_integration_04
```

自检覆盖坐标单位、居中/偏移亮度证据、暗/平坦/极小目标、背景环标注排除、唯一匹配、重复和错类别、crowd IOA、偏移方向、尺寸混杂控制以及图像聚类 bootstrap 可重复性。4 图随机 checkpoint 的推理链路检查严格 EMA 加载、3 层 decoder 及 encoder 导出、官方 COCO 评估、CSV/报告生成。**这些都属于合成验证，不能用它们判断 SAR 假设是否成立。**

另外检查了同一 identity 恢复、已有结果目录拒绝 fresh run、改动协议/图像后的恢复拒绝，以及缺失 EMA 时拒绝评估而非静默回退。对应本地命令是 `python -X utf8 -S outputs/evidence_negative_checks.py`（使用上述环境解释器）。方向未定义的零偏移对象不会被排除出普通中心误差分析；方向关联单列其样本范围。

服务器标准环境无需本地 bootstrap：`python tools/sar_evidence_audit.py --self-test --output NEW_DIRECTORY` 即可。核心依赖是仓库已有 torch/torchvision、numpy、Pillow、PyYAML、scipy、faster-coco-eval。绘图为可选功能：若希望输出 PNG 且服务器未安装 matplotlib，运行 `python -m pip install matplotlib`。CSV/JSON/报告不会因缺少 matplotlib 而停止。

## 3. Git transfer

交付分支：`codex/sar-evidence-validation`。精确 commit 见本次对话最终交付。不要把未跟踪的 `AGENTS.md` 或 `outputs/` 加入提交。

在服务器先查看状态；若有本地改动或分支分叉，先检查，不覆盖：

```bash
cd /home/zxy/sar/repos/rtdetrv2_pytorch
git status --short
git fetch origin
git switch codex/sar-evidence-validation
git pull --ff-only origin codex/sar-evidence-validation
git log -1 --oneline
conda activate rtdetr_zxy
```

## 4. Server launch

数据路径沿用现有配置，而非新增猜测：

```text
images: /home/zxy/sar/datasets/OGSOD-1.0/sar
train:  /home/zxy/sar/datasets/OGSOD-1.0/sar/RTDETR_COCO/train.json
val:    /home/zxy/sar/datasets/OGSOD-1.0/sar/RTDETR_COCO/val.json
```

先做 32 图 smoke，检查数据/权重能加载、query 和诊断输出正常。若脚本报路径不一致，需要先确认真实路径，不创建虚假的目录或用别的数据替代。smoke 不作为论文结论。

```bash
cd /home/zxy/sar/repos/rtdetrv2_pytorch
conda activate rtdetr_zxy
bash tools/run_sar_evidence_audit.sh smoke /home/zxy/sar/experiments/ogsod_r18_baseline_evidence_v1_20261008_smoke32
```

smoke 完成后再启动全验证集：

```bash
cd /home/zxy/sar/repos/rtdetrv2_pytorch
conda activate rtdetr_zxy
bash tools/run_sar_evidence_audit.sh run /home/zxy/sar/experiments/ogsod_r18_baseline_evidence_v1_20261008
```

入口先用 CPU 预检完整 checkpoint/数据/配置，再通过 `CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2` 启动推理，`nohup` 保存日志和 launcher PID，随后 `disown`。不会创建训练任务或新的 detector checkpoint。目录已有内容时拒绝 fresh run。全验证集推理完成后释放分布式通信组，再由 rank 0 计算 CPU 统计，避免另一 rank 在长时间统计期间等待 NCCL。

## 5. Monitoring and metrics

先令 `OUT` 指向所启动的目录；以下示例是全验证集：

```bash
OUT=/home/zxy/sar/experiments/ogsod_r18_baseline_evidence_v1_20261008
tail -n 100 -f "$OUT/console.log"
watch -n 2 nvidia-smi
ps -fp "$(cat "$OUT/launcher.pid")"
python -m json.tool "$OUT/weight_load.json"
python -m json.tool "$OUT/runtime.json"
python -m json.tool "$OUT/analysis_complete.json"
python -c 'import json,sys; d=json.load(open(sys.argv[1])); s=list(d["official_COCO_metrics"])[-1]; m=d["official_COCO_metrics"][s]["stats"]; print("stage",s,"AP",m[0],"AP50",m[1],"AP75",m[2]); print("GT",d["GT_count"],"validity",d["proxy_status_counts"],"failures",d["failure_counts"])' "$OUT/diagnostics.json"
cat "$OUT/report.md"
```

JSON 文件在对应阶段完成后才出现，尚未出现不表示运行失败。统计阶段 GPU 使用率会降低，检查进程和 console.log 的 `Image evidence` 进度。

本次只读诊断不产生 TensorBoard 事件；原训练曲线可这样查看：

```bash
tensorboard --logdir /home/zxy/sar/experiments/ogsod_rtdetrv2_r18_80e_baseline_gbs64_gpu2_zxy/summary --host 0.0.0.0 --port 6006
```

## 6. Outputs and resume

全部新输出位于 `/home/zxy/sar/experiments/ogsod_r18_baseline_evidence_v1_20261008`：

- `report.md`：中文诊断表和判读标准。
- `diagnostics.json`：官方每层 AP、偏相关、CI、有效率、失效分布、敏感性检查。
- `gt_evidence.csv`：各 GT/证据分位数的特征、漏检类型及几何误差。
- `prediction_background.csv`：各分数阈值的 TP/FP 类型和背景统计。
- `stratified_diagnostics.png`：类别/尺寸/来源内证据偏移分组图，需 matplotlib。
- `identity.json`、`weight_load.json`、`runtime.json`：配置、输入、加载和环境证据。
- `inference/queries_rank*/image_ID.npz`：各层查询、类别 logits、精确输出 pair IDs、GT。
- `inference/eval_complete.json`、`inference/*_coco_eval.pth`：官方 COCO 结果。
- `analysis_complete.json`：全部诊断完成标记。
- `console.log`、`launcher.pid`：运行日志与 PID；CPU 预检位于相邻 `.preflight` 目录。

基线输入 `best.pth` 保持不变。本次不生成 `last.pth` 或新训练权重。

中断后确认旧进程已退出，再用相同代码/权重/协议/进程数恢复。未完成的推理阶段会重跑；已完成推理会复用查询，重新完成统计。已完成全部分析时只验证 identity，不覆盖结果。

```bash
cd /home/zxy/sar/repos/rtdetrv2_pytorch
conda activate rtdetr_zxy
bash tools/run_sar_evidence_audit.sh resume /home/zxy/sar/experiments/ogsod_r18_baseline_evidence_v1_20261008
```

smoke 恢复则将上面的目录换为 `/home/zxy/sar/experiments/ogsod_r18_baseline_evidence_v1_20261008_smoke32`，入口自动恢复其 32 图范围。resume 日志使用时间戳，保留原 console.log。

## 7. Caveats

本地没有真实 OGSOD 数据和用户的训练权重，无法从开发环境直接 SSH 到服务器。**当前仅完成工具与合成链路验证，真实实验尚需服务器执行。** exact training commit/启动命令/预训练初始化来源尚未提供，不能声称训练 provenance 完整。现有数据根目录仍由服务器预检确认。

文件名不交叉不保证相邻切片独立；本轮没有做源大图或近重复泄漏检查。无来源 metadata 时不支持跨传感器结论。类别是否桥梁/港口/储罐以实际 COCO categories 为准，不泛化到数据中不存在的目标。

匹配成功的定位分析可能存在选择偏差，因此必须同时查看全 GT 的几何上限和漏检率。全查询最佳 IoU 是 oracle 几何上限，不可写成最终检测召回。背景候选可能有未标注对象；需要人工抽样复核。灰度亮度统计受显示压缩、拉伸和非线性映射影响，不能替代 SLC 成像机理验证。

判断能否继续做模块：主阈值的关联是否清楚、控制尺寸/类别后是否存在、备用分位数方向是否稳定、定位误差是否确实朝证据偏移，并结合失效所在阶段。若关联消失或只出现在稀疏类别，应缩小或更换动机。**不能凭本轮相关性宣称机制因果成立或承诺 AP 提升。**
