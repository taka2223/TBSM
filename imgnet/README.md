# ImageNet-1K Training and Evaluation

This directory contains the ImageNet-1K training and evaluation code for TBSM, with configurations for both pixel- and latent-space generators. Run the commands below from this directory.

## Setup

Install the dependencies:

```bash
pip install -r requirements.txt
```

Download ImageNet-1K and set the local asset locations before selecting a YAML configuration:

```bash
export DM_MODEL_ZOO=path/to/your/model_zoo
export DM_DATA_ZOO=path/to/your/data_zoo
export IMAGENET_DIR=path/to/your/imagenet-1k
# Required only for latent-space configurations:
export IMAGENET_LATENTS_256_DIR=path/to/your/imagenet-latents/sdvae_f8c4/train_256
export IMAGENET_LATENTS_512_DIR=path/to/your/imagenet-latents/sdvae_f8c4/train_512
```

The generator initializations and frozen representation checkpoints used by the provided TBSM configurations are available in the public [TBSM Hugging Face repository](https://huggingface.co/sp12138sp/TBSM). Download them into `$DM_MODEL_ZOO/tbsm_release`:

```bash
huggingface-cli download sp12138sp/TBSM --local-dir "$DM_MODEL_ZOO/tbsm_release"
(cd "$DM_MODEL_ZOO/tbsm_release" && sha256sum -c SHA256SUMS)
```

The Hub repository provides the source mapping and SHA-256 manifest for every hosted asset. External datasets and model assets remain subject to their upstream terms.

For latent-space DiT experiments, download the SD-VAE assets separately from the official [UCGM model files](https://huggingface.co/sp12138sp/UCGM/tree/main), then prepare ImageNet latents with UCGM's [data pipeline](https://github.com/LINs-lab/UCGM/blob/main/data.py) and its ImageNet launchers for [256 × 256](https://github.com/LINs-lab/UCGM/blob/main/scripts/data/in1k256.sh) or [512 × 512](https://github.com/LINs-lab/UCGM/blob/main/scripts/data/in1k512.sh), using `sdvae_f8c4`. The resulting `.safetensors` shards contain `latents`, `latents_flip`, and `labels`, matching this repository's latent-data loader. The supplied latent YAML files expect the VAE at `$DM_MODEL_ZOO/pretrained/sdvae/sd-vae-ft-ema`.

## Train

```bash
bash scripts/train.sh ./configs/xxx/xxx.yaml
```

## Evaluate

Set `evaluation.checkpoint` in the selected YAML file to the checkpoint to evaluate, then run:

```bash
bash scripts/eval.sh ./configs/xxx/xxx.yaml
```

Training uses `runtime.nproc_per_node` from the YAML configuration. Evaluation uses eight GPUs by default; override it when needed, for example with `NPROC_PER_NODE=1`.

The evaluation uses the same six frozen representation models as [FD-Loss](https://github.com/Jiawei-Yang/FD-Loss): Inception, MAE, DINOv2, CLIP, SigLIP2, and ConvNeXtV2. Its repository also provides the corresponding evaluation assets and reference statistics. Place the FD statistics under `$DM_DATA_ZOO/fid_stats` and the ImageNet validation features for precision and recall under `$DM_DATA_ZOO/imagenet-val-prc`. The evaluation script loads Hugging Face and timm models from the local `$DM_MODEL_ZOO` cache by default.

All generator initializations and frozen representation checkpoints referenced by the provided training configurations are hosted in the public [TBSM Hugging Face repository](https://huggingface.co/sp12138sp/TBSM). Latent-space runs additionally require the SD-VAE assets described above.

## Release compatibility

This release contains only the ImageNet-1K TBSM training and evaluation path. The previous low-resolution datasets, non-TBSM methods, visualization/analysis utilities, and their configurations have been removed. Backbone modules now live under `networks/backbones/{dit,jit,pixeldit}/`; use the current YAML configurations rather than earlier module paths or symbols.
