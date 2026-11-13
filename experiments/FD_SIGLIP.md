# FD-SigLIP baseline

This baseline post-trains the same SiT-L/2 initialization used by the TBSM and
SigReg runs. It uses one frozen `siglip2-base-patch16-256` representation and a
differentiable Frechet objective over mean-pooled patch tokens.

The real reference uses all ImageNet training images with the same Dhariwal
256 center crop recorded by the latent-cache manifest. Generated population
moments are initialized from 50,000 one-step samples, then updated with EMA
beta 0.999. Historical moments are detached; gradients flow only through the
current generated batch. Moment and eigendecomposition arithmetic is FP64.

## Reference statistics

```bash
python experiments/compute_fd_siglip_stats.py \
  --root "$IMAGENET_DIR" \
  --index "$IMAGENET_MOMENTS_DIR/sources.json" \
  --model "$SIGLIP2_MODEL_DIR" \
  --output "$FD_SIGLIP_STATS_PATH" \
  --batch 128 --workers 8
```

The `.npz` output is a local data artifact and must not be committed.

## Background training

```bash
export TBSM_PYTHON=/path/to/python
export CUDA_VISIBLE_DEVICES=0
bash experiments/run_fd_siglip.sh start
bash experiments/run_fd_siglip.sh status
```

For a bounded preflight, pass the final step as the second argument:

```bash
bash experiments/run_fd_siglip.sh start 20
```

The run saves full resumable state only as `latest.pt`, retained EMA weights at
20k/40k/60k/80k/100k, FID@10k at those boundaries, and uses the 100k EMA for
the final FID@50k evaluation.
