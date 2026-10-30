# Method description

Team: GIST   Track(s): 1 (this repository; Track 2: https://github.com/maengjemo/2027-EURS-track2)   Reported testing-phase submission: #19149 (composite 0.666)

## Approach
Two independent ensembles at the scoring size (longest side 1024 px): one for segmentation, one for restoration.

**Segmentation: 3 DINOv3 teachers, softmax-averaged.** Each is a frozen DINOv3 ViT-L/16 (`facebook/dinov3-vitl16-pretrain-lvd1689m`)
with a trainable DeBA adapter (dim 64) in the attention blocks, LoRA (q, v, r 16), a DPT decoder (256 ch) and a 5-class head.
Two of them are conditioned on the water type and depth parsed from the file name (FiLM on the decoder output features).
1. combined teacher: 16 EURS scenes (split v1) + 5 external datasets (below), water/depth FiLM;
2. DeBA teacher: all 20 EURS scenes, no conditioning;
3. combined teacher on all 20 scenes + 5 external datasets + pseudo labels of the combined teacher (confidence >= 0.7) on the
   development-phase degraded images (6 development scenes), water/depth FiLM.

**Restoration: 5 physics-constrained models, pixel-averaged.** Each uses the revised underwater image formation model
I = A exp(-beta z) J + B (1 - exp(-beta z)): the channel constants (A, B, beta) come from a (water type, depth) lookup table fitted on EURS
(stored inside the checkpoint) plus a small learned correction, a NAFBlock encoder predicts a low-resolution relative-distance map z
(16 px grid), the model is inverted analytically and the result is refined by a small NAFBlock residual stack (8 blocks).
Variants: split v1 trained twice, all 20 scenes, all 20 scenes wide, all 20 scenes + re-degraded external pairs.

## Training data
- EURS training set: split v1 = 16 training scenes / 4 validation scenes (fish12, fish20, sharkFish, turtles1) for model selection;
  "all 20" models were trained on all 20 scenes with the recipe chosen on split v1.
- Development-phase images (6 development scenes, no ground truth): used only for teacher 3, as pseudo labels of teacher 1.
- External data (disclosed with "Other" on the leaderboard), remapped to the 5 EURS classes:
  Coralscapes, UIIS10K, USIS10K, SUIM, USIS16K (segmentation teachers 1 and 3; restoration model 5 uses their images re-degraded with
  the RSUIGM model fitted on EURS as synthetic pairs).
- Pretrained models: DINOv3 ViT-L/16 (Meta, DINOv3 License, included as `LICENSE_DINOv3.md` in the weights archive).
- No testing-phase image was used for training, fine-tuning, pseudo-labelling or adaptation.
- Recipes: segmentation 20 epochs, 1024x1024 crops, scale 0.5-2.0, batch 2 x 4 accumulation, AdamW (LoRA/adapter 1e-4, decoder 6e-4),
  CE + presence loss, class weights (1, 1, 2, 4, 3), bf16. Restoration 100 epochs, 384x384 crops, batch 4, lr 5e-4, bf16.
  Full configs are stored inside each checkpoint (`config`).

## Anything non-standard at inference
- Segmentation TTA: scales 0.75 / 1.0 / 1.25 x horizontal flip, softmax mean per teacher, then mean over the 3 teachers, argmax.
- Restoration: x8 flip/rot90 self-ensemble per model, 8-bit rounding, mean of the 5 models, PIL `UnsharpMask(radius=1, percent=50, threshold=0)`.
- Water type and depth are read from the trailing `_<type>_<level>` of the file name (allowed by the challenge).
- Restored images are written as JPEG quality 95 (4:4:4), as in the uploaded submission.

## Reproduction notes
- `run.py`: `main(model_dir, input_dir, output_dir, device)`. No network.
- Own `Dockerfile` at the repository root (PyTorch 2.6.0, CUDA 12.4): the DINOv3 classes of `transformers==5.20.2` that our checkpoints
  were saved with need PyTorch >= 2.6 (older transformers versions use a different module layout and cannot load the weights).
  Tested the organizers' way: `--network none --read-only`, unprivileged user, weights mounted read-only at `/pkg/weights`.
  On 12 test images the masks agree with the leaderboard masks on 99.997-100% of pixels and the restored images differ by
  0.07-0.25 grey levels on average (GPU arithmetic: leaderboard RTX 5090 vs test TITAN RTX).
- Leaderboard run: RTX 5090, PyTorch 2.11, fp32. Expect small GPU-dependent differences.
- Run time on one RTX 3090-class GPU: about 15-20 s per image (6 TTA views x 3 ViT-L teachers + 8 views x 5 restoration models; measured 16 s on a shared TITAN RTX), about 5-6 h for 1,200 images.
  GPU memory under 12 GB.
- Training code (not needed for reproduction): private development repository, available on request.
