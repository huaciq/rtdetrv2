"""RT-DETR dataset wrapper that attaches cached whole-image Teacher scores."""
from pathlib import Path

import numpy as np
import torch

from ...core import register
from .coco_dataset import CocoDetection


@register()
class MSARTeacherCocoDetection(CocoDetection):
    """COCO-format dataset with one cached [C] Teacher score per image."""

    def __init__(self, img_folder, ann_file, transforms, teacher_cache_dir=None,
                 return_masks=False, remap_mscoco_category=False):
        super().__init__(img_folder, ann_file, transforms, return_masks,
                         remap_mscoco_category)
        if teacher_cache_dir is None:
            raise ValueError("teacher_cache_dir is required for MSARTeacherCocoDetection")
        cache = Path(teacher_cache_dir)
        self.teacher_scores = np.load(cache / "teacher_scores.fp32.npy", mmap_mode="r")
        self.teacher_ids = np.load(cache / "image_ids.npy", allow_pickle=False)
        expected_classes = len(self.coco.getCatIds())
        if self.teacher_scores.ndim != 2 or self.teacher_scores.shape[1] != expected_classes:
            raise ValueError(
                f"expected teacher scores [N,{expected_classes}], "
                f"got {self.teacher_scores.shape}"
            )
        if len(self.teacher_ids) != len(self.teacher_scores):
            raise ValueError("teacher image_ids and scores have different lengths")
        self.teacher_lookup = {str(image_id): i for i, image_id in enumerate(self.teacher_ids)}

    def __getitem__(self, idx):
        image, target = super().__getitem__(idx)
        image_info = self.coco.loadImgs(self.ids[idx])[0]
        teacher_image_id = str(image_info.get("teacher_image_id", image_info["id"]))
        if teacher_image_id not in self.teacher_lookup:
            raise KeyError(f"missing Teacher cache entry for image {teacher_image_id}")
        cache_index = self.teacher_lookup[teacher_image_id]
        target["teacher_global_score"] = torch.from_numpy(
            np.asarray(self.teacher_scores[cache_index], dtype=np.float32).copy()
        )
        return image, target
