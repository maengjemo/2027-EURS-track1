"""Frozen segmentation teacher on the student's augmented RGB crop."""

from contextlib import nullcontext

import torch
from torch.nn import functional as F

from eurs.models.teacher_restore import metadata_vector
from eurs.models.teacher_seg import load_teacher_seg
from tools.predict_teacher_seg import predict_logits_batch


class OnlineTeacher:
    """Run a checkpoint teacher in small inference chunks to bound peak VRAM.

    ``model`` is an injectable inference module for tests. Student datasets currently
    provide unnormalized RGB in [0, 1], so no ImageNet normalization is applied.
    """

    def __init__(self, ckpt=None, device="cuda", dtype=torch.bfloat16, tta=False,
                 model=None, microbatch=1):
        if model is None and ckpt is None:
            raise ValueError("OnlineTeacher requires ckpt or model")
        self.device = torch.device(device)
        self.model = (model if model is not None else load_teacher_seg(ckpt, self.device))
        if getattr(self.model, "input_mode", "rgb") != "rgb":
            raise ValueError("Online KD requires an RGB teacher checkpoint")
        self.model = self.model.to(self.device).eval().requires_grad_(False)
        self.dtype = {"bf16": torch.bfloat16, "bfloat16": torch.bfloat16,
                      "fp32": torch.float32, "float32": torch.float32}.get(dtype, dtype)
        self.tta = bool(tta)
        self.microbatch = int(microbatch)
        if self.microbatch < 1:
            raise ValueError("microbatch must be positive")

    @torch.no_grad()
    def logits(self, batch_images_01, water, depth, output_hw=None):
        if batch_images_01.ndim != 4 or batch_images_01.shape[1] != 3:
            raise ValueError("Expected RGB batch (B,3,H,W)")
        images = batch_images_01.to(device=self.device, dtype=torch.float32)
        target_hw = tuple(output_hw or images.shape[-2:])
        meta = (metadata_vector(water, depth, device=self.device)
                if getattr(self.model, "meta_cond", False) else None)
        pieces = []
        for start in range(0, len(images), self.microbatch):
            end = start + self.microbatch
            context = (torch.autocast("cuda", dtype=self.dtype)
                       if self.device.type == "cuda" and self.dtype != torch.float32 else nullcontext())
            with context:
                # Same replicate-padding and optional TTA as predict_teacher_seg.py.
                part = predict_logits_batch(self.model, images[start:end], self.device,
                                            tta=self.tta,
                                            meta=meta[start:end] if meta is not None else None)
            if part.shape[-2:] != target_hw:
                part = F.interpolate(part.float(), size=target_hw, mode="bilinear", align_corners=False)
            pieces.append(part.float())
        result = torch.cat(pieces, dim=0)
        if result.shape != (len(images), 5, *target_hw):
            raise ValueError(f"Teacher returned invalid logits shape: {tuple(result.shape)}")
        return result

    @torch.no_grad()
    def logits_and_features(self, batch_images_01, water, depth, output_hw=None):
        """Return logits and backbone stages from one forward per microbatch."""
        if self.tta:
            raise ValueError("feature_kd does not support online_kd.tta")
        if batch_images_01.ndim != 4 or batch_images_01.shape[1] != 3:
            raise ValueError("Expected RGB batch (B,3,H,W)")
        images = batch_images_01.to(device=self.device, dtype=torch.float32)
        target_hw = tuple(output_hw or images.shape[-2:])
        meta = (metadata_vector(water, depth, device=self.device)
                if getattr(self.model, "meta_cond", False) else None)
        logit_parts = []
        feature_parts = {stride: [] for stride in (4, 8, 16, 32)}
        for start in range(0, len(images), self.microbatch):
            end = start + self.microbatch
            h, w = images.shape[-2:]
            padded = F.pad(images[start:end], (0, (-w) % 32, 0, (-h) % 32), mode="replicate")
            context = (torch.autocast("cuda", dtype=self.dtype)
                       if self.device.type == "cuda" and self.dtype != torch.float32 else nullcontext())
            with context:
                output = (self.model(padded, meta=meta[start:end], return_features=True)
                          if meta is not None else self.model(padded, return_features=True))
            logits = output["seg_logits"][..., :h, :w].float()
            if logits.shape[-2:] != target_hw:
                logits = F.interpolate(logits, size=target_hw, mode="bilinear", align_corners=False)
            logit_parts.append(logits)
            stages = output["encoder_features"]
            if len(stages) != 4:
                raise ValueError("feature_kd requires four teacher stages")
            for stride, stage in zip(feature_parts, stages):
                feature_parts[stride].append(stage.detach())
        result = torch.cat(logit_parts, dim=0)
        if result.shape != (len(images), 5, *target_hw):
            raise ValueError(f"Teacher returned invalid logits shape: {tuple(result.shape)}")
        return result, {stride: torch.cat(parts, dim=0) for stride, parts in feature_parts.items()}
