# Paired ImageNet experiment

Both arms use one seed (0), SiT-L/2, SD-VAE EMA, SigLIP2-B/16-256,
lambda=1, rho=0.9, and the original TBSM tracker and training losses.
SIGReg replaces the sampled scattering field with 2048 random projections
of the complete flattened token representation. Projection directions are
averaged into the supervision vector, not supplied to the tracker.

## Generator input interpolation

Both arms use `t_sampling: [1.0, 0.5]`. The unchanged `TBSM.step` samples
one time per example with equal probability and constructs
`x_t = (1-t) * real_latent + t * noise`. Generator and tracker both receive
that time and the class condition. The real latent is sampled from cached
VAE posterior moments and scaled by 0.18215 before interpolation.

This adopts the released **512-resolution** configuration's time schedule
at resolution 256, explicitly requested to retain input interpolation.
The released 256-resolution latent configuration uses `[1.0]` instead,
which leaves the interpolation code present but feeds only pure noise.

The generated source remains an independent draw from the current generator
at `t=1`; EMA evaluation also uses pure noise at `t=1`. These are the original
repository behaviors. Source time need not equal projectile time.
SiT's native time is `1-t`, handled by `UniversalAdapter`; the tracker uses
TBSM time directly. Input interpolation is separate from feature corridor
queries: lambda=1 makes the latter query equal to the projectile feature.

## Portable path configuration

The YAML files contain no machine-specific absolute paths. Set the following
environment variables before launching an ImageNet experiment:

```bash
export IMAGENET_MOMENTS_DIR=path/to/imagenet256_sdvae_ema
export SD_VAE_DECODER_PATH=path/to/sd-vae-ft-ema
export INCEPTION_PATH=path/to/inception-2015-12-05.pkl
export IMAGENET_REF_STATS_PATH=path/to/img256_ref_stats.pkl
export IMM_ROOT=path/to/IMLE/imm
```

The SiT checkpoint is expected at `data/pretrained/sit/SiT-L-2-256.pt`.
`google/siglip2-base-patch16-256` is resolved through Hugging Face; replace it
with a local model directory in the YAML when running offline. Outputs remain
under the ignored `outputs/` directory.

## Run protocol

- One seed (0), 100,000 optimizer updates per arm. Batch size 32, no gradient
  accumulation or activation checkpointing. BF16 autocast with FP32 weights,
  Adam states, feature statistics and SIGReg projection/kernel calculations.
- Every 5,000 updates: EMA sample grid, 16 fixed noises, classes
  `[42, 281, 517, 867]` with four samples per class. Initial step-0 grid too.
- Every 20,000 updates: EMA FID@10k, exactly 10 samples per class, fixed seed 0.
  Uses local StyleGAN Inception weights and the existing ImageNet-256 reference
  statistics. Accumulates moments in FP64 with population covariance, matching
  the IMM reference convention; generated images are converted to uint8 by
  truncation as in IMM's SDVAE decoder. No classifier-free guidance.
- Atomic latest checkpoint every 1,000 updates and at process boundaries,
  containing model, both optimizers, feature statistics, RNG states, SIGReg
  direction counter and data position. Historical EMA weights retained at
  every 20k evaluation point. Only the latest full optimizer checkpoint is kept.
- Independent GPU assignments: energy continues on GPU 1. On 2026-09-19,
  sampled SIGReg on GPU 0 was stopped at the user's request and replaced by a
  fresh SIGReg-IMQ run in `tbsm_sigreg_imq_gpu0`. Sampled SIGReg's last logged
  update is 5710; its complete step-5000 checkpoint and sample grid are retained.
  IMQ starts from the same pretrained SiT initialization with seed 0, not from
  the sampled SIGReg checkpoint. Steps count updates within each arm.

`experiments/run_imagenet_arm.py --arm energy --gpu 1` and
`experiments/run_imagenet_arm.py --arm sigreg --gpu 0` are the independent
runner entry points. The old alternating pair launcher is retired.
Outputs: `outputs/imagenet_sit_l_siglip256/{energy,sigreg}_seed0/`.
The original `TBSM.step`, feature normalization, corridor rule, tracker,
loss self-normalization and optimizer updates are reused.

## Preflight evidence

`experiments/preflight_imagenet.py` tests 512, 256, 128, 64, 32, ... in fresh
processes. The larger batches OOM in the energy arm; 32 passes for both arms
with five complete forward/backward/Adam updates and finite nonzero gradients.
Energy peak allocated/reserved: 58.97/62.18 GiB. SIGReg: 59.12/62.25 GiB.
Physical GPU 1 has 79.33 GiB usable memory. A minimum 5 GiB headroom is required.
Detailed measurements are in `outputs/imagenet_sit_l_siglip256/preflight/`.

The fixed SIGReg field multiplier is **43052.067763388004**, calibrated once
on 32 initial events (four batches of eight) to match energy-field RMS.
Raw SIGReg RMS is about 2.72e-5 versus energy RMS 1.17 in 196,608 dimensions.
The multiplier remains constant throughout training, so it changes field scale,
not its zeros. Bandwidth is 1.0, direction count 2048, projection chunk 128.
Projection RNG is private and checkpointed; it does not advance the global
training RNG. Calibration uses separate directions and discarded model state.

`experiments/imagenet_tbsm.py check` verifies kernel gradient sign, explicit
projection chunk sums, reference cancellation, RNG isolation and replay,
the unchanged energy target, SiT time conversion, and data-order resumption.
Preflight FID on 32 images is only an execution check, not an FID@10k result.

## Deliberate differences from the released 256 latent configuration

SiT-L/2 replaces UCGM DiT-XL/2, and SigLIP2 alone is used as agreed. The
pretrained SiT null-class embedding is retained for strict weight loading but
training class dropout is disabled to keep triplets class conditional.
Lambda=1, rho=0.9; source generation uses the current model, evaluation its EMA.
The existing SD-VAE posterior moment cache supplies mean/std draws before the
0.18215 latent scaling. It contains center crops without separately encoded
horizontal flips; neither arm adds approximate latent-space flips. This is
a controlled kernel comparison, not an exact reproduction of the paper's FID.

## Analytic direction average ablation

`sigreg_imq.yaml` uses `field.type: sigreg_imq`. It replaces sampled directions
with the high-dimensional IMQ approximation to the direction-averaged Gaussian
kernel. For flattened feature dimension `d`, displacement `delta = reference - h`,
and bandwidth `ell`, the pair force is

```text
b(h, reference) = delta / (d * ell^2)
                  * (1 + ||delta||^2 / (d * ell^2))^(-3/2)
v(h; r, s) = scale * (b(h, r) - lambda * b(h, s))
```

This is the high-dimensional approximation described here, not the exact
finite-dimensional hypergeometric formula. It uses the full `256 * 768` feature
dimension, FP32 kernel arithmetic under BF16 autocast, and no direction RNG.
Label computation takes O(Bd); reference sampling and network costs remain.
The two-moons implementation is unchanged: the high-dimensional approximation
is not an exact direction integral in 2D.

The config preserves seed 0, bandwidth 1, scale **43052.067763388004**, batch 32,
tracker/time inputs, input interpolation, normalization, optimizer settings,
and the 5k visualization / 20k FID@10k schedule. Scale is deliberately inherited
without recalibration to isolate the field-estimator change. Removing direction
noise can change label RMS and effective gradients under loss self-normalization;
equal scale does not guarantee equal actual optimizer updates.

Outputs are isolated in `outputs/imagenet_sit_l_siglip256/sigreg_imq_seed0/`.
Start a fresh run for this ablation; do not resume a sampled-direction checkpoint.
The existing paired preflight script still targets energy and sampled SIGReg;
it does not select an IMQ batch size. The inherited batch is a starting point,
not a new measured memory result.

CPU verification (no pretrained assets or GPU required):

```bash
python experiments/test_sigreg_imq.py
python experiments/imagenet_tbsm.py check
```

Launched on GPU 0 on 2026-09-19 with
`experiments/run_imagenet_arm.py --arm sigreg_imq --gpu 0`, in tmux session
`tbsm_sigreg_imq_gpu0`. Runner status and training history live in the isolated
output directory above; the energy run continues independently on GPU 1.
