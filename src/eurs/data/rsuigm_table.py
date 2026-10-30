"""(water, depth) → RSUIGM channel-constant lookup table.

The forward degradation ``I_c = A_c exp(-beta_c z) J_c + B_c (1-exp(-beta_c z))``
has channel constants ``(A_c, B_c, beta_c)`` that are almost fully determined by
the water type ``w`` and depth ``d`` (per-water K_d IQR <= 0.004 in our analysis),
while the spatial variation is carried by the single relative-distance map ``z``.

This module turns the per-image fit products (``rsuigm_fit.jsonl``) plus the
water-level K_d summary (``rsuigm_summary.json``) into a dense ``(10, 20, 9)``
table indexed by ``(water 0..9, depth index)`` with 9 channels laid out as
``A(R,G,B), B(R,G,B), beta(R,G,B)``. ``TeacherPhysRestore`` copies this table into
a buffer so it travels inside the checkpoint and the server never needs the jsonl.

Conventions (match the rest of the repo):
- The fit jsonl and image filenames use raw water ``1..10`` and raw depth metres.
- The model / ``metadata_vector`` convention uses water ``0..9``; this module's
  public API (``lookup``) takes that ``0..9`` convention and maps to raw ``w+1``.
- ``_is_black_condition`` (from ``redegrade``) marks the fully-dark conditions that
  are excluded from fitting.
"""

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from .redegrade import DEFAULT_FIT_PATH, _is_black_condition

DEFAULT_SUMMARY_PATH = DEFAULT_FIT_PATH.parent / "rsuigm_summary.json"

N_WATER = 10
DEPTHS = list(range(5, 101, 5))  # 20 depth bins, metres
N_DEPTH = len(DEPTHS)
N_CHANNEL = 9  # A(RGB), B(RGB), beta(RGB)

# Valid ranges used to clamp table entries (also the fallback placeholder values).
A_RANGE = (0.0, 1.2)
B_RANGE = (0.0, 1.0)
BETA_MIN = 1e-3

# Placeholder values used only via the fallback statistics below when some
# conditions have no fits. The full-placeholder case (no data at all) is reached
# only with ``require=False`` on the checkpoint-load path, where
# ``load_state_dict`` overwrites the buffer so these numbers never train a model.
_PLACEHOLDER = np.array([0.5, 0.5, 0.5, 0.1, 0.1, 0.1, 1.0, 1.0, 1.0], dtype=np.float32)
_PLACEHOLDER_ZMEAN = 1.0


def depth_index(depth):
    """Map a depth in metres to its bin index in ``DEPTHS`` (nearest, clamped)."""
    idx = torch.round(torch.as_tensor(depth, dtype=torch.float32) / 5.0) - 1
    return idx.long().clamp(0, N_DEPTH - 1)


def _clamp_const(values):
    """Clamp a length-9 constant vector into the valid A/B/beta ranges."""
    values = np.asarray(values, dtype=np.float32).copy()
    values[0:3] = np.clip(values[0:3], *A_RANGE)
    values[3:6] = np.clip(values[3:6], *B_RANGE)
    values[6:9] = np.maximum(values[6:9], BETA_MIN)
    return values


class RSUIGMTable:
    """Dense (water, depth) → (A, B, beta) lookup built from the RSUIGM fits.

    Build priority, per ``(water, depth)``:
      1. Median A/B/beta over the fitted images at exactly that condition.
      2. Otherwise A from the K_d summary ``A_c(d) = A0_c * exp(-K_c d)``; B from
         the nearest fitted depth within the same water; beta from the per-water
         median (or the global median as a last resort).

    All entries are clamped to the A/B/beta ranges. ``z_mean`` holds the per-water
    median of the fitted z-map means (used by the model to initialise the z grid).

    ``bounds_lo``/``bounds_hi`` are the per-channel sigmoid bounds for the nometa
    regression head; they span the full ``(water, depth)`` range of the table (not
    just the per-water median), so every table value is representable.

    ``allowed_scenes`` filters fits before all medians and regression. When it is
    supplied, K/A0 are recomputed from the selected scenes; the global summary
    is deliberately ignored because it may include validation GT. With no
    filter, the provided summary's kd_table is used for backward compatibility.

    ``require=True`` (default, new-training path) fails if the fit jsonl has no
    usable selected fits, or if the needed K/A0 source is empty. ``require=False``
    is only for checkpoint loading, where ``load_state_dict`` restores the table.
    """

    def __init__(self, fit_jsonl=DEFAULT_FIT_PATH, summary_json=DEFAULT_SUMMARY_PATH,
                 require=True, allowed_scenes=None):
        self.fit_jsonl = Path(fit_jsonl)
        self.summary_json = Path(summary_json)
        self.allowed_scenes = None if allowed_scenes is None else frozenset(allowed_scenes)
        if self.allowed_scenes is not None and not self.allowed_scenes:
            raise ValueError("allowed_scenes must contain at least one training scene")
        by_cond = defaultdict(lambda: {"A": [], "B": [], "beta": [], "z": []})
        by_water = defaultdict(lambda: {"A": [], "B": [], "beta": [], "z": []})
        by_scene_water = defaultdict(list)
        if self.fit_jsonl.exists():
            self._read_fits(by_cond, by_water, by_scene_water)
        # The published summary includes validation GT fits. Never read its
        # kd_table when a scene filter is active: recompute K and A0 solely from
        # the same training fits used for the condition medians.
        kd_table = (self._read_summary() if self.allowed_scenes is None
                    else self._fit_kd_table(by_scene_water))
        # ``require=True`` is the new-training path: the physics table must come
        # from the real analysis products, so missing/empty files fail loudly
        # instead of silently filling placeholder constants. ``require=False`` is
        # only for the checkpoint-load path, where ``load_state_dict`` overwrites
        # the buffer right after and the fit jsonl need not exist on the server.
        if require:
            self._require_sources(by_cond, by_water, kd_table)

        self.table = np.empty((N_WATER, N_DEPTH, N_CHANNEL), dtype=np.float32)
        self.z_mean = np.empty((N_WATER,), dtype=np.float32)
        global_z = [z for agg in by_water.values() for z in agg["z"]]
        global_z_med = float(np.median(global_z)) if global_z else _PLACEHOLDER_ZMEAN
        global_beta = [b for agg in by_water.values() for b in agg["beta"]]
        global_beta_med = (np.median(np.stack(global_beta), axis=0)
                           if global_beta else _PLACEHOLDER[6:9])
        global_a = [a for agg in by_water.values() for a in agg["A"]]
        global_b = [b for agg in by_water.values() for b in agg["B"]]
        global_a_med = np.median(np.stack(global_a), axis=0) if global_a else _PLACEHOLDER[0:3]
        global_b_med = np.median(np.stack(global_b), axis=0) if global_b else _PLACEHOLDER[3:6]

        for w in range(N_WATER):
            raw = w + 1  # raw filename water type
            water_agg = by_water.get(raw)
            self.z_mean[w] = (float(np.median(water_agg["z"]))
                              if water_agg and water_agg["z"] else global_z_med)
            water_beta_med = (np.median(np.stack(water_agg["beta"]), axis=0)
                              if water_agg and water_agg["beta"] else global_beta_med)
            for di, d in enumerate(DEPTHS):
                self.table[w, di] = self._cell(raw, d, by_cond, by_water, kd_table,
                                               water_beta_med, global_a_med, global_b_med)

        # Sigmoid bounds for the no-metadata regression head. These must span the
        # full per-channel range the table actually takes over *all* (water, depth)
        # conditions. The earlier per-water median (median over depth) collapsed the
        # strong depth decay of A_c(d)=A0 exp(-K d): e.g. A_R at w=1,d=5 is ~0.83
        # while the median-over-depth upper bound was ~0.19, so the regression head
        # could never reach shallow-depth constants. Taking min/max over the raw
        # table guarantees every lookup value is representable by the sigmoid.
        flat = self.table.reshape(-1, N_CHANNEL)  # (water*depth, 9)
        lo = flat.min(axis=0)
        hi = flat.max(axis=0)
        span = np.maximum(hi - lo, 1e-3)
        self.bounds_lo = lo.astype(np.float32)
        self.bounds_hi = (lo + span).astype(np.float32)
        # Keep the beta lower bound strictly positive so nometa regression stays valid.
        self.bounds_lo[6:9] = np.maximum(self.bounds_lo[6:9], BETA_MIN)
        self.bounds_hi[6:9] = np.maximum(self.bounds_hi[6:9], self.bounds_lo[6:9] + 1e-3)
        # The sigmoid maps onto the open interval, so pad the bounds slightly to
        # keep the extreme table values strictly inside (they stay representable).
        pad = 0.02 * span.astype(np.float32)
        self.bounds_lo = self.bounds_lo - pad
        self.bounds_hi = self.bounds_hi + pad
        self.bounds_lo[0:3] = np.maximum(self.bounds_lo[0:3], A_RANGE[0])
        self.bounds_hi[0:3] = np.minimum(self.bounds_hi[0:3], A_RANGE[1])
        self.bounds_lo[3:6] = np.maximum(self.bounds_lo[3:6], B_RANGE[0])
        self.bounds_hi[3:6] = np.minimum(self.bounds_hi[3:6], B_RANGE[1])
        self.bounds_lo[6:9] = np.maximum(self.bounds_lo[6:9], BETA_MIN)
        # Every table entry must fall within the regression bounds by construction.
        assert np.all(self.table.reshape(-1, N_CHANNEL) >= self.bounds_lo - 1e-6)
        assert np.all(self.table.reshape(-1, N_CHANNEL) <= self.bounds_hi + 1e-6)

    def _read_fits(self, by_cond, by_water, by_scene_water):
        with self.fit_jsonl.open(encoding="utf-8") as source:
            for line in source:
                row = json.loads(line)
                if self.allowed_scenes is not None and row.get("scene") not in self.allowed_scenes:
                    continue
                if not row.get("fit_ok"):
                    continue
                water, depth = int(row["w"]), int(row["d"])
                if _is_black_condition(water, depth):
                    continue
                try:
                    a = np.asarray(row["A"], dtype=np.float32)
                    b = np.asarray(row["B"], dtype=np.float32)
                    beta = np.asarray(row["beta_rel"], dtype=np.float32)
                except KeyError:
                    continue
                if a.shape != (3,) or b.shape != (3,) or beta.shape != (3,):
                    continue
                if not (np.isfinite(a).all() and np.isfinite(b).all() and np.isfinite(beta).all()):
                    continue
                for store in (by_cond[(water, depth)], by_water[water]):
                    store["A"].append(a)
                    store["B"].append(b)
                    store["beta"].append(beta)
                by_scene_water[(row.get("scene"), water)].append((depth, a))
                stats = row.get("z_stats") or {}
                mean = stats.get("mean")
                if mean is not None and np.isfinite(mean):
                    by_cond[(water, depth)]["z"].append(float(mean))
                    by_water[water]["z"].append(float(mean))

    def _require_sources(self, by_cond, by_water, kd_table):
        """Fail loudly when the analysis products needed for a real table are absent.

        Called only on the new-training path (``require=True``). A table built
        purely from placeholder constants would train on wrong physics without any
        error, so this checks both source files exist and carry usable content.
        """
        if not self.fit_jsonl.exists():
            raise FileNotFoundError(
                f"RSUIGM fit jsonl not found: {self.fit_jsonl}. Pass require=False "
                "only on the checkpoint-load path (buffer is overwritten there).")
        if self.allowed_scenes is None and not self.summary_json.exists():
            raise FileNotFoundError(
                f"RSUIGM summary json not found: {self.summary_json}. Pass "
                "require=False only on the checkpoint-load path.")
        usable = any(agg["A"] for agg in by_cond.values())
        if not usable:
            raise ValueError(
                f"RSUIGM fit jsonl {self.fit_jsonl} has no usable (non-black, "
                "finite) fits for the selected training scenes; cannot build a physics table.")
        if not kd_table:
            raise ValueError(
                "RSUIGM training fits or summary have an empty kd_table; "
                "cannot build depth-extrapolated constants.")

    @staticmethod
    def _fit_kd_table(by_scene_water):
        """Reproduce the summary's per-scene log-A slope, using selected fits only."""
        slopes = defaultdict(lambda: [[], [], []])
        intercepts = defaultdict(lambda: [[], [], []])
        for (_, water), rows in by_scene_water.items():
            depth = np.asarray([d for d, _ in rows], dtype=np.float64)
            a = np.stack([v for _, v in rows]).astype(np.float64)
            for channel in range(3):
                valid = (a[:, channel] > 0.01) & (a[:, channel] < 1.5)
                # Match summarize(): each scene/water regression needs at
                # least four fitted conditions.
                if valid.sum() < 4:
                    continue
                x, y = depth[valid], np.log(a[valid, channel])
                xm, ym = x.mean(), y.mean()
                vx = np.mean((x - xm) ** 2)
                slope = np.mean((x - xm) * (y - ym)) / vx if vx >= 1e-12 else 0.0
                slopes[water][channel].append(-slope)
                intercepts[water][channel].append(ym - slope * xm)
        return {str(water): {
            "K": [float(np.median(values)) if values else float("nan")
                  for values in channels],
            "A0": [float(np.exp(np.median(values))) if values else float("nan")
                   for values in intercepts[water]],
        } for water, channels in slopes.items()}

    def _read_summary(self):
        if not self.summary_json.exists():
            return {}
        data = json.loads(self.summary_json.read_text(encoding="utf-8"))
        return data.get("kd_table", {})

    def _cell(self, raw, d, by_cond, by_water, kd_table, water_beta_med, global_a_med, global_b_med):
        agg = by_cond.get((raw, d))
        if agg and agg["A"]:
            a = np.median(np.stack(agg["A"]), axis=0)
            b = np.median(np.stack(agg["B"]), axis=0)
            beta = np.median(np.stack(agg["beta"]), axis=0)
            return _clamp_const(np.concatenate([a, b, beta]))

        # Fallback. A from the K_d summary when available.
        entry = kd_table.get(str(raw))
        if entry is not None and "A0" in entry and "K" in entry:
            a0 = np.asarray(entry["A0"], dtype=np.float32)
            k = np.asarray(entry["K"], dtype=np.float32)
            a = a0 * np.exp(-np.nan_to_num(k, nan=0.0) * d)
            a = np.where(np.isfinite(a), a, global_a_med)
        else:
            a = global_a_med

        # B from the nearest fitted depth within the same water, else per-water,
        # else global median.
        b = self._nearest_depth_median(raw, d, by_cond, "B")
        if b is None:
            agg_w = by_water.get(raw)
            if agg_w and agg_w["B"]:
                b = np.median(np.stack(agg_w["B"]), axis=0)
            else:
                b = global_b_med
        beta = water_beta_med
        return _clamp_const(np.concatenate([np.asarray(a), np.asarray(b), np.asarray(beta)]))

    @staticmethod
    def _nearest_depth_median(raw, target_d, by_cond, key):
        """Median ``key`` constant at the fitted depth closest to ``target_d``."""
        candidates = [(abs(dd - target_d), dd, agg) for (ww, dd), agg in by_cond.items()
                      if ww == raw and agg[key]]
        if not candidates:
            return None
        _, _, agg = min(candidates, key=lambda item: item[0])
        return np.median(np.stack(agg[key]), axis=0)

    def lookup(self, water, depth):
        """Return per-sample A/B/beta torch tensors for water (0..9) and depth (m).

        ``water`` and ``depth`` may be python scalars or 1-D tensors/arrays of the
        same length. Returns a dict with ``A``, ``B``, ``beta`` each shaped
        ``(batch, 3)``.
        """
        table = torch.from_numpy(self.table)
        water = torch.as_tensor(water, dtype=torch.long).reshape(-1).clamp(0, N_WATER - 1)
        depth = torch.as_tensor(depth, dtype=torch.float32).reshape(-1)
        if water.numel() != depth.numel():
            raise ValueError("water and depth must have the same length")
        di = depth_index(depth)
        values = table[water, di]  # (batch, 9)
        return {"A": values[:, 0:3], "B": values[:, 3:6], "beta": values[:, 6:9]}
