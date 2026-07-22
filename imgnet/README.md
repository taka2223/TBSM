# ImageNet-1K Training and Evaluation

This directory contains the ImageNet-1K training and evaluation code for TBSM, with configurations for both pixel- and latent-space generators. Run the commands below from this directory.

## Setup

Install the dependencies:

```bash
pip install -r requirements.txt
```

Download ImageNet-1K and the required pretrained models, then replace the `/path/to/your/...` entries in the selected YAML configuration. Set the shared model and evaluation-data directories when needed:

```bash
export DM_MODEL_ZOO=/path/to/your/model_zoo
export DM_DATA_ZOO=/path/to/your/data_zoo
```

For latent-space DiT experiments, the initialization backbones are released as TiT-XL/2 at 256 × 256 and TiT-XL/4 at 512 × 512 in [UCGM](https://github.com/LINs-lab/UCGM). Download the corresponding multi-step TiT checkpoints and SD-VAE assets from the official [UCGM model files](https://huggingface.co/sp12138sp/UCGM/tree/main), then prepare ImageNet latents with UCGM's [data pipeline](https://github.com/LINs-lab/UCGM/blob/main/data.py) and its ImageNet launchers for [256 × 256](https://github.com/LINs-lab/UCGM/blob/main/scripts/data/in1k256.sh) or [512 × 512](https://github.com/LINs-lab/UCGM/blob/main/scripts/data/in1k512.sh), using `sdvae_f8c4`. The resulting `.safetensors` shards contain `latents`, `latents_flip`, and `labels`, matching this repository's latent-data loader. Point `experiment.init_weights`, `latent.decoder_path`, and `data.data_dir` in the selected YAML file to the corresponding downloaded and generated assets.

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

The JiT and PixelDiT generator initializations can be downloaded directly from the official [JiT](https://github.com/LTH14/JiT) and [PixelDiT](https://github.com/NVlabs/PixelDiT) repositories; latent-space DiT assets follow the UCGM TiT workflow above. All pretrained models used in our experiments originate from official releases. For convenience, we will also release all checkpoints required by the provided training configurations.
