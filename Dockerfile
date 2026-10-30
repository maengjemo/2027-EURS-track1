# Own image (allowed by the template): transformers 5.20.2 (DINOv3 ViT classes used by our checkpoints) needs PyTorch >= 2.6,
# the base image has 2.4.0. CUDA 12.4 runtime, Python 3.11.
FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt
ENV HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
COPY . /pkg
WORKDIR /pkg
