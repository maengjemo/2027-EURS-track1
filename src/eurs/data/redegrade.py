"""RSUIGM parameter sampling and forward re-degradation of paired GT images."""

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image


DEFAULT_FIT_PATH = Path("/mnt/SSD2/eurs/analysis_out/rsuigm_fit.jsonl")


def _is_black_condition(water, depth):
    return (water == 10 and depth >= 65) or (water == 9 and depth >= 95)


class RSUIGMParamBank:
    """Successful per-image fits, grouped by the original water/depth condition."""

    def __init__(self, path=DEFAULT_FIT_PATH):
        self.path = Path(path)
        self.by_condition = defaultdict(list)
        self.z_stats_by_scene_condition = {}
        with self.path.open(encoding="utf-8") as source:
            for line in source:
                row = json.loads(line)
                if not row.get("fit_ok"):
                    continue
                water, depth = int(row["w"]), int(row["d"])
                if _is_black_condition(water, depth):
                    continue
                arrays = ("A", "B", "beta_rel")
                if any(np.asarray(row.get(key, [])).shape != (3,) for key in arrays):
                    continue
                if not all(np.isfinite(row[key]).all() for key in arrays):
                    continue
                stats = row.get("z_stats") or {}
                if not all(key in stats and np.isfinite(stats[key]) for key in ("mean", "std")):
                    continue
                params = {"scene": row["scene"], "w": water, "d": depth,
                          "A": np.asarray(row["A"], dtype=np.float32),
                          "B": np.asarray(row["B"], dtype=np.float32),
                          "beta_rel": np.asarray(row["beta_rel"], dtype=np.float32),
                          "z_stats": stats}
                self.by_condition[(water, depth)].append(params)
                self.z_stats_by_scene_condition[(row["scene"], water, depth)] = stats
        self.conditions = sorted(self.by_condition)
        if not self.conditions:
            raise ValueError(f"No usable RSUIGM fits in {self.path}")
        self.zmap_paths = sorted(self.path.parent.glob("rsuigm_zmap_*.npz"))

    def __len__(self):
        return sum(map(len, self.by_condition.values()))

    def sample(self, rng, water=None, depth=None):
        """Select a condition uniformly, then one fitted image within that condition."""
        conditions = [key for key in self.conditions
                      if (water is None or key[0] == int(water))
                      and (depth is None or key[1] == int(depth))]
        if not conditions:
            raise ValueError(f"No usable RSUIGM fit for water={water}, depth={depth}")
        condition = conditions[int(rng.integers(len(conditions)))]
        rows = self.by_condition[condition]
        return rows[int(rng.integers(len(rows)))]

    def sample_zmap_path(self, rng):
        if not self.zmap_paths:
            return None
        return self.zmap_paths[int(rng.integers(len(self.zmap_paths)))]


def make_zmap(h, w, z_stats, rng, zmap_path=None):
    """Generate a smooth relative-distance map with fitted per-image moments.

    An NPZ source must contain the ``z`` array. If absent, a 4x8 random field is
    bilinearly resized. The weak vertical gradient is randomized in direction.
    """
    if h <= 0 or w <= 0:
        raise ValueError("z map dimensions must be positive")
    if zmap_path is not None:
        with np.load(zmap_path) as source:
            coarse = np.asarray(source["z"], dtype=np.float32)
        if coarse.ndim != 2 or not np.isfinite(coarse).all():
            raise ValueError(f"Invalid z map in {zmap_path}")
    else:
        coarse = rng.normal(size=(4, 8)).astype(np.float32)
    field = np.array(Image.fromarray(coarse, mode="F").resize((w, h), Image.Resampling.BILINEAR),
                     dtype=np.float32, copy=True)
    if zmap_path is None:
        field += np.linspace(-0.5, 0.5, h, dtype=np.float32)[:, None] * float(rng.uniform(-0.5, 0.5))
    mean = max(0.0, float(z_stats["mean"]))
    std = max(0.0, float(z_stats["std"]))
    field = (field - field.mean()) / max(float(field.std()), 1e-6)
    field = mean + std * field
    lower = max(0.0, float(z_stats.get("min", 0.0)))
    upper = max(lower, float(z_stats.get("max", max(mean + 4 * std, lower))))
    return np.clip(field, lower, upper).astype(np.float32)


def redegrade(gt_img, params, zmap, rng):
    """I = A exp(-beta*z) J + B (1-exp(-beta*z)); beta_G is one."""
    gt = np.asarray(gt_img, dtype=np.float32)
    z = np.asarray(zmap, dtype=np.float32)
    if gt.ndim != 3 or gt.shape[-1] != 3 or z.shape != gt.shape[:2]:
        raise ValueError("Expected GT (H,W,3) and z map (H,W)")
    a = np.asarray(params["A"], dtype=np.float32)
    b = np.asarray(params["B"], dtype=np.float32)
    beta = np.asarray(params["beta_rel"], dtype=np.float32)
    if a.shape != (3,) or b.shape != (3,) or beta.shape != (3,):
        raise ValueError("A, B and beta_rel must have three channels")
    beta = beta.copy()
    beta[1] = 1.0
    transmission = np.exp(-np.maximum(beta, 0)[None, None, :] * np.maximum(z, 0)[..., None])
    result = a[None, None, :] * transmission * gt + b[None, None, :] * (1 - transmission)
    return np.clip(result, 0, 1).astype(np.float32)
