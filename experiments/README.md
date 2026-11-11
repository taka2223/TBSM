# Two moons: energy distance versus SIGReg in TBSM

`two_moons_tbsm.py` is a standalone two-dimensional instance of the TBSM
frozen-target training algorithm. It does not change the ImageNet or MNIST code.

## Objective and controls

- Target: an equal mixture of two semicircles, with independent Gaussian noise
  of standard deviation 0.05 in each coordinate. Fresh population samples are
  generated at every step. Neither moon identity nor paired latent targets are
  given to the generator.
- Generator: `G(z) = z + MLP(z)`, `z ~ N(0, I_2)`, three hidden layers of width
  64, SiLU, zero-initialized output layer. All arms start from the same Gaussian.
- Representation: identity. No per-sample normalization, changing whitening
  statistics, pretrained representation, generator EMA, or diffusion path.
- Scattering: lambda = 1; one independent real source and generated source per
  event. `--references R` averages R independent events per projectile. R=1
  is the original single-triplet estimator; larger R changes variance and cost,
  not the population field.
- Energy field: `(r-h)/||r-h|| - (s-h)/||s-h||`.
- SIGReg field: uniform unit directions `a` in 2D, Gaussian frequency weight
  `t ~ N(0, ell^-2)`. Integrating frequency analytically gives
  `k(h,r) = E_a exp(-(a dot (h-r))^2 / (2 ell^2))`. Each step resamples 32
  directions. The derivative is computed analytically; sources are detached.
- Tracker: the same width-64, three-hidden-layer SiLU MLP for both kernels,
  with two-dimensional vector output. It predicts the **full averaged field**,
  as in the released TBSM vector tracker. It is not the scalar projected-force
  network proposed in `note.md`.
- Generator target: `stopgrad(h + (1-rho)*v_sample + rho*v_tracker)`.
  Generator loss: `0.5 * mean(sum((h-target)^2))`; tracker loss is MSE against
  the sampled field. Both predictions/losses are formed before either update.
  Source draws are independent of the query and current tracker history.
- Compare rho=0 and rho=0.9 for each kernel. Use identical data/noise RNG
  streams across arms, and a separate direction stream for SIGReg.

The experiment uses the mathematical frozen-target loss, without the released
image code's per-step `loss / stopgrad(loss)` normalization or sqrt(feature_dim)
field scale. These controls make the toy objective and gradient explicit.
It is not a reproduction of the paper's image training hyperparameters.

## Run

Dependencies: PyTorch, NumPy, SciPy, Matplotlib. No dataset download required.

```bash
# Check field signs, frozen-target gradients, and the Gaussian frequency integral.
python experiments/two_moons_tbsm.py --check

# Main suite: 30k updates, B=1024, 1 source pair per query, ell=0.15,
# Adam G lr=2e-4 / tracker lr=1e-3, shared cosine decay to 5% of initial lr.
# Single seed (0), four arms; only the selected physical GPU is visible.
CUDA_VISIBLE_DEVICES=1 python experiments/run_two_moons_suite.py

# Variance reduction diagnostic, four arms, seed 0.
CUDA_VISIBLE_DEVICES=1 python experiments/run_two_moons_suite.py \
  --references 16 --seeds 0 --output outputs/two_moons_tbsm

# An individual run (CPU also supported).
python experiments/two_moons_tbsm.py --method sigreg --rho 0.9 \
  --device cuda --bandwidth 0.15 --references 16 --batch 1024 \
  --steps 30000 --eval-every 3000 --lr-decay cosine

# Recreate figures and summary from completed runs.
python experiments/two_moons_tbsm.py --aggregate outputs/two_moons_single_reference --aggregate-seed 0
```

Use the Python environment in which this repository and its dependencies are installed.

## Evaluation

Intermediate metrics use 1,024 fixed, independently drawn validation latents
and real samples. Final metrics use **fresh** 4,096 generated and 4,096 real
samples, with 128 new projection directions. Each seed also evaluates a second
independent real sample against the same reference (real-versus-real baseline).

- **SW2:** square root of average one-dimensional empirical Wasserstein-2
  squared over projection directions. This is an empirical two-sample metric;
  even real-versus-real is positive.
- **Energy U:** unbiased two-sample energy-distance estimator; negative finite
  sample estimates are possible and are retained.
- **SIGReg MMD² U:** an unbiased estimate for the actual SIGReg kernel. In 2D,
  its angular average is `exp(-q) I0(q)`, `q=||x-y||²/(4 ell²)`. Evaluation
  integrates **both** angle and frequency analytically, using SciPy's stable
  `i0e`, rather than reusing the training projections.
- **RBF MMD² U:** independent diagnostic averaged over ordinary 2D Gaussian
  kernels with bandwidths 0.1, 0.25, 0.5 and 1.0. This differs from SIGReg's
  projected kernel and helps detect missing fine structure.
- **Off manifold:** fraction farther than 0.15 (3 noise standard deviations)
  from the closest noiseless moon centerline. Measured for real samples too.
- **Within 1 sigma / median centerline distance:** diagnostics of local
  thickness. A sharp collapsed arc plus a few bridging samples can have a
  small global discrepancy but the wrong local noise distribution. The real
  within-1-sigma fraction is approximately 68%, measured rather than assumed.
- **Upper mass:** fraction assigned to the upper moon by nearest centerline;
  should be near 0.5.
- **Arc TV:** total variation between the empirical nearest-arc histogram
  (32 equal-angle bins on each moon) and uniform mass across 64 bins. Noise,
  nearest-curve projection and finite samples produce a nonzero baseline.

`history.json` contains convergence traces; `samples.npz` includes snapshots
and independent final samples; `checkpoint.pt` contains the generator/tracker;
`config.json` records settings; `result.json` contains final metrics. Aggregation
writes `distributions.png`, `convergence.png`, `evolution.png`, `geometry.png`, `summary.csv`
and `summary.json`. Results are exploratory: an approximate match on one toy
distribution is not a neural convergence theorem or evidence for image scale.

References: [TBSM](https://arxiv.org/html/2607.18198v1),
[LeJEPA / SIGReg](https://arxiv.org/html/2511.08544), and the local `note.md`.

## Completed experiment

The [single-seed report](../outputs/two_moons_single_reference/report.md) contains
the results and figures. Under the standard single-source setting, final SW2
was 0.0249 for SIGReg + tracker versus 0.0624 for energy + tracker; an independent
real-versus-real comparison was 0.0240. Local thickness and off-manifold mass
still differ from the target, so this supports approximate matching rather than
exact distributional convergence. All scripts default to seed 0 only.
