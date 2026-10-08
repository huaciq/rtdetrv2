# D-FINE-R18：同预算定位基线可行性

这是源码与合成数据检查记录，**不是 OGSOD 实验结果**。没有训练或评估真实验证集，也没有验证 D-FINE 的 OGSOD 定位优势。

## 官方源码与比较边界

官方仓库：https://github.com/Peterande/D-FINE；固定 commit `956d1709314c2c6a4df6f34de232054578a7449f`。

该版本提供 HGNetv2 配置，没有现成官方 D-FINE-R18 配置或对应官方检测 checkpoint。本次仅组合官方已经注册的 PResNet18、HybridEncoder、DFINETransformer，不新增模块、不修改官方源码。R18 是本项目适配基线，不能称为官方发布的 R18 benchmark。

| 项目 | 原 E0 | 准备的 D-FINE-R18 pilot | 比较含义 |
|---|---|---|---|
| backbone | PResNet18 vd | 同一 PResNet18 vd | 同一权重格式、输出 stride/channel |
| 初始化 | ImageNet ResNet18_vd | 同一个下载 URL | 不用 COCO/Objects365、E0 best 或 E4 权重 |
| 输入、数据 | 固定640×640、现有 train/val | 相同 | 原 Resize 是分轴缩放，不是 letterbox |
| encoder 通道/超参 | 128/256/512 → hidden256；E0设置 | 同一组数值设置 | 内部拓扑不同，不能称为同一 encoder |
| encoder 拓扑 | CSPRepLayer | 官方 RepNCSPELAN4 | D-FINE 原生结构差异保留并披露 |
| decoder | 3层、300 queries、100 denoising | 同样；4/4/4采样点 | 保留各自的原生 refinement |
| query ranking、matcher cost 配置 | 原设置 | 共享对应数值设置 | D-FINE criterion 的跨层 union regression matching 仍不同 |
| loss | VFL1、bbox5、GIoU2 | 同权重 + 官方 FGL0.15、DDF1.5 | 原生 FDR/GO-LSD、LQE、pre-box监督；不是纯 FDR ablation |
| optimizer | AdamW，4e-4/backbone4e-5 | 相同 | 相同 betas、weight decay、clip、warmup |
| augmentation/EMA/AMP | E0 | 同一配置 | D-FINE collate 关闭自身 multiscale，stop_epoch 保留在 pilot 以外 |
| 时间预算 | 新训练前20 epochs | 新训练20 epochs | 两边 seed0、global batch64、2×3090；不比较旧80e best |
| LR schedule | E0 milestone67/warmup1000 | 相同 | 前20e截断；不为某一模型单独优化短程schedule |

这可作为 **backbone/预算/数据匹配的完整检测器比较**，不能据此单独证明 distribution refinement 是原因。官方 native D-FINE-S 的 HGNetv2、多尺度/强增强、132e、stage2 best reload/EMA restart 均不直接拿来与旧 E0 比较；本 pilot 不进入 stage2。

主要证据：[PResNet](https://github.com/Peterande/D-FINE/blob/956d1709314c2c6a4df6f34de232054578a7449f/src/nn/backbone/presnet.py)、[encoder](https://github.com/Peterande/D-FINE/blob/956d1709314c2c6a4df6f34de232054578a7449f/src/zoo/dfine/hybrid_encoder.py)、[criterion](https://github.com/Peterande/D-FINE/blob/956d1709314c2c6a4df6f34de232054578a7449f/src/zoo/dfine/dfine_criterion.py)、[训练阶段](https://github.com/Peterande/D-FINE/blob/956d1709314c2c6a4df6f34de232054578a7449f/src/solver/det_solver.py)。

两边初始化 URL：`https://github.com/lyuwenyu/storage/releases/download/v0.1/ResNet18_vd_pretrained_from_paddle.pth`。本地构造检查关闭下载；证明 state 格式兼容，不等于已下载和核验服务器预训练文件。服务器使用同一缓存文件并记录 SHA256，见执行指南。

## 本地实际检查（2026-10-08）

Windows，conda `pytorch`，torch2.5.1/torchvision0.20.1、numpy1.26.4、faster-coco-eval1.6.6、pycocotools2.0.11。D-FINE 导入补充 calflops0.3.2、loguru0.7.3、accelerate1.10.1；transformers4.57.6 已存在。未升级 torch/torchvision/numpy。

| 检查 | E0 pilot | D-FINE pilot |
|---|---:|---:|
| 全部参数（3 classes） | 20,085,596 | 20,002,661 |
| 可训练参数 | 20,085,596 | 20,002,659 |
| 输入 | [1,3,640,640] | 相同 |
| backbone输出 | [1,128,80,80] / [1,256,40,40] / [1,512,20,20] | 相同 |
| 预测框/类别 | [1,300,4] / [1,300,3] | 相同 |
| postprocess | 300框，labels∈{0,1,2} | 相同 |
| native DN/aux loss（合成两GT） | 有限 | 有限 |
| detector state严格保存/加载 | 通过 | 通过 |
| EMA state严格保存/加载 | 通过 | 通过 |

将 E0 的合成初始化 backbone state 严格载入 D-FINE，在完全相同输入下，state SHA256 和三层 feature 数值字节 SHA256 全部相同。未做 backward/optimizer.step；未测试真实数据、服务器 AMP/DDP、GPU训练时间或推理速度。loss 值仅用于数值检查，不用于模型效果判断。

## 固定一次 pilot 的判定

同时比较20e的原生 best AP checkpoint，并建议记录 epoch19 last 的同固定预算指标，防止只看 best 的选择差异。只比较新 pilot 对：AP、AP75、APs、Bridge AP75、Storage Tank AP75；记录参数、各自完整训练时间、同GPU固定batch1/640/FP32的 model+postprocessor latency。没有统计显著性或多seed证据时，差异只能作为下一步验证信号。

两边采用同一个 `localization_pilot_check.py` 的 canonical COCO 全验证评估，top300导出、COCO maxDets=100/类别、EMA严格加载、不额外NMS/score threshold。保留各自原生 COCO evaluator，不修改 baseline。

D-FINE 若 AP75/两困难类别 AP75 提升而 APs 无下降，可认为完整强定位基线值得进一步验证；若仅AP50提高、分类别结果不一致或代价明显增加，不能宣称已解决小目标高IoU定位。20e不优于E0也不能排除较长训练下的效果；本任务不继续延长预算或 sweep。

配置、数据 preflight、训练/恢复/评估命令见 `LOCALIZATION_EVIDENCE_GUIDE.md`。当前结论：**工程上可行；OGSOD 定位优势待实验验证。**
