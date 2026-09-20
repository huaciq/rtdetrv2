import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch

from src.core import YAMLConfig


cfg = YAMLConfig(
    "configs/rtdetrv2/rtdetrv2_r18vd_120e_msar_gkd.yml",
    PResNet={"pretrained": False},
)
loader = cfg.train_dataloader
samples, targets = next(iter(loader))
model = cfg.model.to("cuda")
criterion = cfg.criterion.to("cuda")
samples = samples.cuda()
targets = [{key: value.cuda() for key, value in target.items()} for target in targets]
with torch.no_grad():
    outputs = model(samples, targets=targets)
losses = criterion(outputs, targets)
print("SMOKE_OK", tuple(samples.shape), tuple(outputs["pred_logits"].shape),
      tuple(targets[0]["teacher_global_score"].shape),
      float(losses["loss_gkd_raw"]), float(losses["loss_gkd"]))
