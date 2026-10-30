"""Track 1 reproduction of testing-phase submission #19149 (team GIST).

The organizers call   main(model_dir, input_dir, output_dir, device).

Pipeline (identical to the leaderboard run, step by step):
  0. each full-resolution input is resized with PIL BILINEAR so that its longest side is 1024 px (aspect kept; the
     organizers' scale_to_max_side rule). All models run at this scoring size.
  1. segmentation: 3 DINOv3 ViT-L/16 teachers (DeBA adapter + DPT head) with test-time augmentation
     (scales 0.75 / 1.0 / 1.25 x horizontal flip, softmax mean, stored as float16 log-probabilities like the
     leaderboard run); the 3 teachers' softmax are averaged and the argmax is the mask (labels 0..4).
  2. restoration: 5 physics-guided restoration models, each with the x8 flip/rot90 self-ensemble, rounded to 8 bit;
     the 5 images are averaged and rounded, then PIL UnsharpMask(radius=1, percent=50, threshold=0).
  3. outputs: seg_masks/<stem>.png (mode L) and restored/<stem>.jpg (JPEG quality 95, 4:4:4 chroma), the same file
     format as the uploaded submission zip.
The water type and depth used by the meta-conditioned models are read from the trailing `_<type>_<level>` of the
file name (stem.rsplit("_", 2)); the scene part of the name is never used.
"""
import os
import sys

import numpy as np
import torch
from PIL import Image, ImageFilter
from torch.nn import functional as F

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from eurs.models.teacher_restore import load_teacher_restore, metadata_vector  # noqa: E402
from eurs.models.teacher_seg import load_teacher_seg  # noqa: E402

MAX_SIDE = 1024
SEG_CKPTS = ("teachers_seg/seg_comb_deba_film_ext5_last.pth",   # combined teacher (16 scenes + 5 external sets)
             "teachers_seg/seg_deba_all20_last.pth",            # DeBA teacher, 20 scenes
             "teachers_seg/seg_comb_devst_all20_last.pth")      # combined teacher, 20 scenes + dev pseudo labels
RESTORE_CKPTS = ("teachers_restore/restore_phys_meta_s0_best.pth",
                 "teachers_restore/restore_phys_meta_s1_best.pth",
                 "teachers_restore/restore_phys_meta_all20_best.pth",
                 "teachers_restore/restore_phys_meta_wide_all20_best.pth",
                 "teachers_restore/restore_phys_meta_ext5rd_all20_best.pth")
TTA_SCALES = (0.75, 1.0, 1.25)
VIEWS = [(f, k) for f in (False, True) for k in range(4)]


def scale_to_max_side(w, h, max_side=MAX_SIDE):
    scale = max_side / max(w, h)
    if scale >= 1:
        return w, h
    return max(1, round(w * scale)), max(1, round(h * scale))


def water_depth(stem):
    _, water, depth = stem.rsplit("_", 2)
    return int(water) - 1, int(depth)


def to_tensor(image):
    return torch.from_numpy(np.asarray(image, dtype=np.float32).transpose(2, 0, 1).copy() / 255)[None]


@torch.no_grad()
def seg_forward(model, tensor, device, meta):
    h, w = tensor.shape[-2:]
    padded = F.pad(tensor, (0, (-w) % 32, 0, (-h) % 32), mode="replicate").to(device)
    out = model(padded, meta=meta) if meta is not None else model(padded)
    return out["seg_logits"][..., :h, :w]


@torch.no_grad()
def seg_tta_logprob(model, tensor, device, meta):
    h, w = tensor.shape[-2:]
    prob_sum, count = None, 0
    for scale in TTA_SCALES:
        scaled = tensor if scale == 1.0 else F.interpolate(
            tensor, size=(max(1, round(h * scale)), max(1, round(w * scale))), mode="bilinear", align_corners=False)
        for flip in (False, True):
            view = torch.flip(scaled, dims=[-1]) if flip else scaled
            logits = seg_forward(model, view, device, meta)
            if flip:
                logits = torch.flip(logits, dims=[-1])
            logits = F.interpolate(logits, size=(h, w), mode="bilinear", align_corners=False)
            prob = F.softmax(logits, dim=1)
            prob_sum = prob if prob_sum is None else prob_sum + prob
            count += 1
    logp = torch.log((prob_sum / count).clamp_min(1e-8))[0]
    return logp.cpu().numpy().astype(np.float16)   # the leaderboard run stored the logits as float16


@torch.no_grad()
def restore_x8(model, tensor, device, meta):
    def one(view):
        h, w = view.shape[-2:]
        padded = F.pad(view.to(device), (0, (-w) % 32, 0, (-h) % 32), mode="replicate")
        return model(padded, meta=meta)["restored_image"][..., :h, :w]
    acc = None
    for flip, k in VIEWS:
        view = torch.flip(tensor, dims=[-1]) if flip else tensor
        view = torch.rot90(view, k, dims=(-2, -1)) if k else view
        out = one(view)
        out = torch.rot90(out, -k, dims=(-2, -1)) if k else out
        out = torch.flip(out, dims=[-1]) if flip else out
        acc = out if acc is None else acc + out
    restored = (acc / len(VIEWS))[0].clamp(0, 1).cpu()
    return np.clip(np.rint(restored.permute(1, 2, 0).numpy() * 255), 0, 255).astype(np.uint8)


def main(model_dir, input_dir, output_dir, device="cuda"):
    torch.manual_seed(0)
    torch.backends.cudnn.benchmark = False
    device = device if (device != "cuda" or torch.cuda.is_available()) else "cpu"
    seg_models = [load_teacher_seg(os.path.join(model_dir, p), device) for p in SEG_CKPTS]
    rest_models = [load_teacher_restore(os.path.join(model_dir, p), device) for p in RESTORE_CKPTS]
    os.makedirs(os.path.join(output_dir, "restored"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "seg_masks"), exist_ok=True)
    names = sorted(f for f in os.listdir(input_dir) if f.lower().endswith((".png", ".jpg", ".jpeg")))
    for i, name in enumerate(names):
        stem = os.path.splitext(name)[0]
        with Image.open(os.path.join(input_dir, name)) as im:
            image = im.convert("RGB")
            target = scale_to_max_side(*image.size)
            if image.size != target:
                image = image.resize(target, Image.Resampling.BILINEAR)
        tensor = to_tensor(image)
        water, depth = water_depth(stem)
        meta = metadata_vector(water, depth, device)

        probs = None
        for model in seg_models:
            m_meta = meta if getattr(model, "meta_cond", False) else None
            logp = torch.from_numpy(seg_tta_logprob(model, tensor, device, m_meta)).to(device).float()
            p = logp.softmax(0)
            probs = p if probs is None else probs + p
        mask = (probs / len(seg_models)).argmax(0).to(torch.uint8).cpu().numpy()
        Image.fromarray(mask, "L").save(os.path.join(output_dir, "seg_masks", f"{stem}.png"))

        imgs = []
        for model in rest_models:
            m_meta = meta if getattr(model, "use_meta", False) else None
            imgs.append(restore_x8(model, tensor, device, m_meta).astype(np.float32))
        mean = np.clip(np.rint(sum(imgs) / len(imgs)), 0, 255).astype(np.uint8)
        sharp = Image.fromarray(mean, "RGB").filter(ImageFilter.UnsharpMask(radius=1, percent=50, threshold=0))
        sharp.save(os.path.join(output_dir, "restored", f"{stem}.jpg"), "JPEG", quality=95, subsampling=0)
        print(f"[{i + 1}/{len(names)}] {stem} {image.size}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4] if len(sys.argv) > 4 else "cuda")
