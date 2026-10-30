"""Frozen physics teacher followed by a pretrained Restormer residual refiner."""

from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from .restormer_arch import Restormer
from .teacher_phys import TeacherPhysRestore


DEFAULT_RESTORMER_CFG = {
    "dim": 48,
    "num_blocks": [4, 6, 6, 8],
    "heads": [1, 2, 4, 8],
    "num_refinement_blocks": 4,
    "LayerNorm_type": "BiasFree",  # official real_denoising.pth uses bias-free norms
}


class TeacherRestormerRefiner(nn.Module):
    def __init__(self, base_ckpt=None, restormer_cfg=None, pretrained_path=None,
                 freeze_base=True, residual_scale=1.0, base_config=None):
        super().__init__()
        if base_ckpt is not None:
            from .teacher_restore import load_teacher_restore
            self.base = load_teacher_restore(base_ckpt)
            if not isinstance(self.base, TeacherPhysRestore):
                raise ValueError("base_ckpt must contain a teacher_phys model")
            base_config = self.base.config_dict()
        elif base_config is not None:
            build = {k: v for k, v in base_config.items() if k != "name"}
            self.base = TeacherPhysRestore(allow_placeholder=True, **build)
        else:
            raise ValueError("base_ckpt or base_config is required")
        self.base_config = dict(base_config)
        self.use_meta = self.base.use_meta
        self.freeze_base = bool(freeze_base)
        if self.freeze_base:
            self.base.requires_grad_(False)
            self.base.eval()
        self.residual_scale = float(residual_scale)
        self.restormer_cfg = {**DEFAULT_RESTORMER_CFG, **(restormer_cfg or {})}
        self.restormer = Restormer(inp_channels=6, out_channels=3, **self.restormer_cfg)
        if pretrained_path is not None:
            self._load_pretrained(pretrained_path)
        # The official Restormer predicts restored RGB with an input skip. A
        # zero gate preserves the proven physics output exactly at initialization
        # while keeping every pretrained Restormer weight intact for fine-tuning.
        self.residual_gate = nn.Parameter(torch.zeros(()))

    def _load_pretrained(self, path):
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"Restormer pretrained weights missing: {path}")
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        state = checkpoint.get("params", checkpoint.get("state_dict", checkpoint))
        state = {k.removeprefix("module."): v for k, v in state.items()}
        key = "patch_embed.proj.weight"
        expected = self.restormer.state_dict()[key]
        if state[key].shape != (expected.shape[0], 3, *expected.shape[2:]):
            raise ValueError(f"Pretrained first convolution has unexpected shape: {state[key].shape}")
        expanded = expected.new_zeros(expected.shape)
        expanded[:, :3] = state[key]
        state[key] = expanded
        self.restormer.load_state_dict(state, strict=True)

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_base:
            self.base.eval()
        return self

    def forward(self, inp, meta=None):
        if self.freeze_base:
            with torch.no_grad():
                base_output = self.base(inp, meta=meta)
        else:
            base_output = self.base(inp, meta=meta)
        j0 = base_output["restored_image"].float()
        h, w = inp.shape[-2:]
        stacked = torch.cat((j0.to(inp.dtype), inp), dim=1)
        stacked = F.pad(stacked, (0, (-w) % 8, 0, (-h) % 8), mode="replicate")
        candidate = self.restormer(stacked)[..., :h, :w]
        residual = candidate.float() - j0
        restored = (j0 + self.residual_scale * self.residual_gate.float() * residual).clamp(0, 1)
        return {**base_output, "restored_image": restored, "affine_image": j0}

    def config_dict(self):
        return {"name": "teacher_restormer", "base_config": self.base_config,
                "restormer_cfg": self.restormer_cfg, "freeze_base": self.freeze_base,
                "residual_scale": self.residual_scale}
