# EURS @ MaCVi WACV 2027: Track 1 reproduction (team GIST, submission #19149)

```bash
python repro_check.py . --download-weights                      # fetch weights/EURS_track1_19149_weights.zip (link in weights/weights_link.yaml)
docker build -t eurs-t1 .                                        # own image: PyTorch 2.6.0 + transformers 5.20.2
docker run --rm --gpus all --network none -v "$PWD/weights":/pkg/weights:ro -v /path/to/degraded:/in:ro -v /tmp/out:/out \
    eurs-t1 python -c 'import run; run.main("/pkg/weights", "/in", "/out", "cuda")'
```
`run.py` writes `restored/<stem>.jpg` and `seg_masks/<stem>.png` at the scoring size. Method and data: `METHOD.md`.
Track 2 (submission #19347): <https://github.com/maengjemo/2027-EURS-track2>. Built from <https://github.com/CeviKle/EURS-MaCVi2027-repro-template>.
