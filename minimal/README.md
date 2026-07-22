# Minimal TBSM

This directory provides a compact MNIST implementation of the standard TBSM training workflow. Before running it, replace the `/path/to/your/...` placeholders in the selected config with your local dataset and checkpoint paths.

Run all commands from this directory:

```bash
cd minimal
```

## Setup

The minimal implementation uses the same Python environment as the ImageNet code. Install the shared dependencies from this directory:

```bash
pip install -r ../imgnet/requirements.txt
```

## Standard TBSM training

Use the frozen ResNet-18 representation model and the default tracked-scattering configuration:

```bash
python main.py --config ./configs/resnet.yaml
```

Set `data.data_dir` to the directory containing MNIST and set `method.params.matchers[0].representor.ckpt_path` to a ResNet-18 checkpoint before running.

## Training without a representation model

Use the identity representation to apply the scattering objective directly in pixel space:

```bash
python main.py --config ./configs/identity.yaml
```

This configuration is useful for testing TBSM without a pretrained representation model.

## Exploring the Figure 3 design map

Edit the following parameters in `./configs/resnet.yaml` to explore the combinations shown in Figure 3 of the paper:

```yaml
params:
  intra_weight: 0.9 # lambda
  track_weight: 0.9 # rho
```

- `intra_weight` is the intra-source weight $\lambda$.
- `track_weight` is the tracked-supervision weight $\rho$.

Both parameters take values in $[0, 1]$. The four reference configurations are:

| $\rho$ | $\lambda$ | Configuration |
|---:|---:|---|
| 0 | 1 | Instant scattering |
| 1 | 1 | Tracked scattering |
| 0 | 0 | Inter-only scattering |
| 1 | 0 | Fake-to-real scattering |

Generated samples and checkpoints are written to the directory specified by `experiment.save_dir`.
