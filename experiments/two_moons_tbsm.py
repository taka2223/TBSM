"""Compare energy-distance and sliced-Gaussian MMD fields in toy TBSM.

Streaming population two moons, independent real/source/query draws.
Run --help for training, numerical checks, and aggregation commands.
"""

import argparse
import csv
import json
import math
import os
from pathlib import Path
import time

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(__file__).resolve().parents[1] / ".runtime" / "matplotlib"),
)
import numpy as np
import torch
from torch import nn
from scipy.spatial.distance import cdist
from scipy.spatial import cKDTree
from scipy.special import i0e


def moons(n, rng, noise=0.05):
    """Balanced population mixture; no reuse of a finite training dataset."""
    angle = torch.rand(n, generator=rng) * math.pi
    lower = torch.rand(n, generator=rng) < 0.5
    x = torch.where(lower, 1 - angle.cos(), angle.cos()) - 0.5
    y = torch.where(lower, 0.5 - angle.sin(), angle.sin()) - 0.25
    return torch.stack((x, y), -1) + noise * torch.randn(n, 2, generator=rng)


def directions(n, rng, dtype=torch.float32):
    angle = torch.rand(n, generator=rng, dtype=dtype) * (2 * math.pi)
    return torch.stack((angle.cos(), angle.sin()), -1)


class MLP(nn.Module):
    def __init__(self, width=64, residual=False):
        super().__init__()
        self.residual = residual
        self.net = nn.Sequential(
            nn.Linear(2, width), nn.SiLU(),
            nn.Linear(width, width), nn.SiLU(),
            nn.Linear(width, width), nn.SiLU(), nn.Linear(width, 2),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x):
        out = self.net(x)
        return out + x if self.residual else out


def field(h, r, s, method, a=None, bandwidth=0.3):
    if method == "energy":
        dr, ds = r - h, s - h
        # Exact bearings except for the measure-zero collision convention.
        return (dr / dr.norm(dim=-1, keepdim=True).clamp_min(1e-12)
                - ds / ds.norm(dim=-1, keepdim=True).clamp_min(1e-12))
    dr, ds = (r - h) @ a.T, (s - h) @ a.T
    inv_l2 = bandwidth ** -2
    forces = (dr * torch.exp(-0.5 * dr.square() * inv_l2)
              - ds * torch.exp(-0.5 * ds.square() * inv_l2)) * inv_l2
    return forces @ a / len(a)


def numerical_checks():
    """Independent kernel differentiation and Fourier quadrature checks."""
    rng = torch.Generator().manual_seed(847)
    h = torch.randn(11, 2, generator=rng, dtype=torch.float64, requires_grad=True)
    r = torch.randn(11, 2, generator=rng, dtype=torch.float64)
    s = torch.randn(11, 2, generator=rng, dtype=torch.float64)
    a = directions(23, rng, torch.float64)
    ell = 0.3
    errors = {}
    for method in ("energy", "sigreg"):
        if method == "energy":
            potential = ((h-r).norm(dim=-1) - (h-s).norm(dim=-1)).sum()
        else:
            kr = torch.exp(-0.5 * ((h-r) @ a.T / ell).square()).mean(-1)
            ks = torch.exp(-0.5 * ((h-s) @ a.T / ell).square()).mean(-1)
            potential = (ks-kr).sum()
        exact = -torch.autograd.grad(potential, h)[0]
        estimated = field(h, r, s, method, a, ell)
        errors[method + "_gradient_max_error"] = (exact-estimated).abs().max().item()
        target = (h + estimated).detach()
        loss = 0.5 * (h-target).square().sum()
        grad = torch.autograd.grad(loss, h)[0]
        assert torch.allclose(grad, -exact, atol=1e-10, rtol=1e-10)
    # The frequency weight is N(0, ell^-2), integrated on +/- 9 std.
    t = torch.linspace(-9 / ell, 9 / ell, 20001, dtype=torch.float64)
    delta = torch.linspace(-3, 3, 37, dtype=torch.float64)
    density = ell / math.sqrt(2 * math.pi) * torch.exp(-0.5 * (ell*t).square())
    quadrature = torch.trapezoid(torch.cos(delta[:, None]*t) * density, t)
    analytic = torch.exp(-0.5*(delta/ell).square())
    errors["fourier_kernel_max_error"] = (quadrature-analytic).abs().max().item()
    angles = torch.arange(65536, dtype=torch.float64) * (2*math.pi/65536)
    angular = torch.exp(-0.5*(delta[:,None]*angles.cos()/ell).square()).mean(-1)
    bessel = torch.from_numpy(i0e(delta.numpy()**2/(4*ell**2)))
    errors["angular_kernel_max_error"] = (angular-bessel).abs().max().item()
    assert max(errors.values()) < 1e-10, errors
    print(json.dumps(errors, indent=2), flush=True)


def centerline_distances(x):
    angle = np.linspace(0, np.pi, 1024)
    upper = np.stack((np.cos(angle)-0.5, np.sin(angle)-0.25), -1)
    lower = np.stack((0.5-np.cos(angle), 0.25-np.sin(angle)), -1)
    curve = np.concatenate((upper, lower))
    return cKDTree(curve).query(x)


def geometry_metrics(x, noise=0.05):
    distance, nearest = centerline_distances(x)
    bins = (nearest//1024)*32 + np.minimum((nearest % 1024)*32//1024, 31)
    histogram = np.bincount(bins, minlength=64)/len(x)
    return dict(off_manifold=float(np.mean(distance > 3*noise)),
                within_1sigma=float(np.mean(distance <= noise)),
                centerline_distance_median=float(np.median(distance)),
                upper_mass=float(np.mean(nearest < 1024)),
                arc_tv=float(0.5*np.abs(histogram-1/64).sum()))


def metrics(x, y, eval_directions, noise=0.05, bandwidth=0.3):
    """Unbiased two-sample energy/MMD, empirical SW2, geometry diagnostics."""
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    n, m = len(x), len(y)
    xx, yy, xy = cdist(x, x), cdist(y, y), cdist(x, y)
    energy = 2*xy.mean() - xx.sum()/(n*(n-1)) - yy.sum()/(m*(m-1))
    mmds = []
    for ell in (0.1, 0.25, 0.5, 1.0):
        kxx = np.exp(-xx**2/(2*ell**2))
        kyy = np.exp(-yy**2/(2*ell**2))
        kxy = np.exp(-xy**2/(2*ell**2))
        mmds.append((kxx.sum()-n)/(n*(n-1)) + (kyy.sum()-m)/(m*(m-1)) - 2*kxy.mean())
    # In two dimensions the uniform angular integral is analytic:
    # E_a exp(-(a.dot(delta))^2/(2*l^2)) = exp(-q)*I0(q), q=||delta||^2/(4*l^2).
    sigreg_mmd = ((i0e(xx**2/(4*bandwidth**2)).sum()-n)/(n*(n-1))
                  + (i0e(yy**2/(4*bandwidth**2)).sum()-m)/(m*(m-1))
                  - 2*i0e(xy**2/(4*bandwidth**2)).mean())
    px, py = np.sort(x @ eval_directions.T, axis=0), np.sort(y @ eval_directions.T, axis=0)
    sw2 = np.sqrt(np.mean((px-py)**2))
    return dict(energy_u=float(energy), rbf_mmd2_u=float(np.mean(mmds)),
                sigreg_mmd2_u=float(sigreg_mmd),
                sw2=float(sw2), **geometry_metrics(x, noise))


def train(args):
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    # Separate RNG streams ensure equal queries and sources across methods.
    data_rng = torch.Generator().manual_seed(10000 + args.seed)
    angle_rng = torch.Generator().manual_seed(20000 + args.seed)
    eval_rng = torch.Generator().manual_seed(30000 + args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is inaccessible; run outside the sandbox with CUDA_VISIBLE_DEVICES=1")
    g, tracker = MLP(args.width, residual=True).to(device), MLP(args.width).to(device)
    go = torch.optim.Adam(g.parameters(), lr=args.gen_lr, betas=(0.9, 0.99))
    to = torch.optim.Adam(tracker.parameters(), lr=args.tracker_lr, betas=(0.9, 0.99))
    run = args.output / f"{args.method}_rho{args.rho:g}_seed{args.seed}"
    run.mkdir(parents=True, exist_ok=True)
    config = {k: str(v) if isinstance(v, Path) else v for k,v in vars(args).items()}
    config["torch_version"] = torch.__version__
    config["device_name"] = torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"
    (run / "config.json").write_text(json.dumps(config, indent=2))
    z_eval = torch.randn(args.eval_n, 2, generator=eval_rng).to(device)
    real_eval = moons(args.eval_n, eval_rng).numpy()
    eval_a = directions(64, eval_rng).numpy()
    logs, snapshots = [], {}
    started = time.monotonic()
    for step in range(args.steps + 1):
        if step % args.eval_every == 0 or step == args.steps:
            with torch.no_grad():
                fake = g(z_eval).cpu().numpy()
            report = metrics(fake, real_eval, eval_a, bandwidth=args.bandwidth)
            report.update(step=step, elapsed=time.monotonic()-started)
            logs.append(report)
            snapshots[f"step_{step}"] = fake
            (run / "history.json").write_text(json.dumps(logs, indent=2))
            print(json.dumps(dict(method=args.method, rho=args.rho, seed=args.seed, **report)), flush=True)
        if step == args.steps:
            break
        if args.lr_decay == "cosine":
            factor = 0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * step / args.steps))
            for group in go.param_groups:
                group["lr"] = args.gen_lr * factor
            for group in to.param_groups:
                group["lr"] = args.tracker_lr * factor
        real = moons(args.batch * args.references, data_rng).to(device).view(args.batch, args.references, 2)
        z = torch.randn(args.batch, 2, generator=data_rng).to(device)
        zs = torch.randn(args.batch * args.references, 2, generator=data_rng).to(device)
        h = g(z)
        with torch.no_grad():
            source = g(zs).view(args.batch, args.references, 2)
            a = directions(args.directions, angle_rng).to(device) if args.method == "sigreg" else None
            label = field(h.detach()[:, None, :], real, source, args.method, a, args.bandwidth).mean(1)
        predicted = tracker(h.detach()) if args.rho > 0 else label
        # Evaluate both losses before either optimizer update, matching TBSM.
        mixed = ((1-args.rho)*label + args.rho*predicted).detach()
        target = (h.detach() + mixed).detach()
        gen_loss = 0.5*(h-target).square().sum(-1).mean()
        go.zero_grad(set_to_none=True)
        gen_loss.backward()
        if args.rho > 0:
            to.zero_grad(set_to_none=True)
            tracker_loss = 0.5*(predicted-label).square().sum(-1).mean()
            tracker_loss.backward()
        go.step()
        if args.rho > 0:
            to.step()
    torch.save(dict(generator=g.state_dict(), tracker=tracker.state_dict()), run / "checkpoint.pt")
    final_rng = torch.Generator().manual_seed(40000+args.seed)
    with torch.no_grad():
        final_fake = g(torch.randn(args.final_n, 2, generator=final_rng).to(device)).cpu().numpy()
    final_real = moons(args.final_n, final_rng).numpy()
    baseline = moons(args.final_n, final_rng).numpy()
    final_a = directions(128, final_rng).numpy()
    final_metrics = metrics(final_fake, final_real, final_a, bandwidth=args.bandwidth)
    baseline_metrics = metrics(baseline, final_real, final_a, bandwidth=args.bandwidth)
    np.savez_compressed(run / "samples.npz", **snapshots, final_fake=final_fake,
                        final_real=final_real, baseline=baseline)
    result = dict(method=args.method, rho=args.rho, seed=args.seed,
                  metrics=final_metrics, real_vs_real=baseline_metrics,
                  elapsed=time.monotonic()-started)
    (run / "result.json").write_text(json.dumps(result, indent=2))
    print("FINAL " + json.dumps(result), flush=True)


def aggregate(root, seed=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    files = sorted(root.glob("*/result.json"))
    if not files:
        raise ValueError(f"No results in {root}")
    groups = {}
    for path in files:
        result = json.loads(path.read_text())
        if seed is not None and result["seed"] != seed:
            continue
        # Recompute cheap geometry diagnostics on saved independent final samples.
        # This also backfills thickness diagnostics for early pilot checkpoints.
        with np.load(path.parent/"samples.npz") as samples:
            result["metrics"].update(geometry_metrics(samples["final_fake"]))
            result["real_vs_real"].update(geometry_metrics(samples["baseline"]))
        path.write_text(json.dumps(result, indent=2))
        key = (result["method"], result["rho"])
        groups.setdefault(key, []).append((path.parent, result))
    keys = sorted(groups)
    if not keys:
        raise ValueError(f"No completed runs for seed {seed}")
    colors = {("energy", 0.0): "#4878D0", ("energy", 0.9): "#123B80",
              ("sigreg", 0.0): "#ECA13E", ("sigreg", 0.9): "#B74619"}
    def label(key):
        return f"{'Energy' if key[0]=='energy' else 'SIGReg'} | rho={key[1]:g}"
    summary = []
    for key in keys:
        values = groups[key]
        row = dict(method=key[0], rho=key[1], seeds=len(values))
        shared_metrics = set.intersection(*(set(r["metrics"]) for _,r in values))
        for metric in sorted(shared_metrics):
            nums = [r["metrics"][metric] for _, r in values]
            row[metric+"_mean"] = float(np.mean(nums))
            row[metric+"_std"] = float(np.std(nums, ddof=1 if len(nums)>1 else 0))
        summary.append(row)
    # Baselines are identical across arms for a seed: count each seed once.
    baseline_by_seed = {r["seed"]: r["real_vs_real"] for values in groups.values() for _,r in values}
    baseline_summary = {m: dict(mean=float(np.mean([r[m] for r in baseline_by_seed.values()])),
                               std=float(np.std([r[m] for r in baseline_by_seed.values()],
                                                 ddof=1 if len(baseline_by_seed)>1 else 0)))
                        for m in next(iter(baseline_by_seed.values()))}
    (root / "summary.json").write_text(json.dumps(dict(runs=summary, baseline=baseline_summary), indent=2))
    with (root / "summary.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    # Final distribution, seed zero (or first available seed).
    fig, axes = plt.subplots(1, len(keys)+1, figsize=(3.2*(len(keys)+1), 3.3), sharex=True, sharey=True)
    real = np.load(groups[keys[0]][0][0] / "samples.npz")["final_real"]
    axes[0].scatter(*real[:2048].T, s=2, alpha=0.4, color="#39774B", rasterized=True)
    axes[0].set_title("Target (fresh samples)")
    for ax,key in zip(axes[1:], keys):
        path,result = groups[key][0]
        samples = np.load(path / "samples.npz")["final_fake"]
        ax.scatter(*samples[:2048].T, s=2, alpha=0.4, color=colors.get(key), rasterized=True)
        ax.set_title(label(key)+f"\nseed={result['seed']}, SW2={result['metrics']['sw2']:.3f}")
    for ax in axes:
        ax.set_aspect("equal"); ax.set_xlim(-2,2); ax.set_ylim(-1.5,1.5)
        ax.grid(alpha=0.15)
    config = json.loads((groups[keys[0]][0][0]/"config.json").read_text())
    fig.suptitle(f"Two moons: one-step neural generators, lambda=1, "
                 f"{config['references']} source pair(s)/query, ell={config['bandwidth']:g}", y=1.02)
    fig.tight_layout(); fig.savefig(root/"distributions.png", dpi=170, bbox_inches="tight"); plt.close(fig)
    fig,axes = plt.subplots(2, 2, figsize=(11,7))
    for ax,metric,title in zip(axes.flat, ("sw2","energy_u","rbf_mmd2_u","off_manifold"),
                               ("Sliced Wasserstein-2", "Energy distance (unbiased)",
                                "RBF MMD squared (unbiased)", "Off manifold (> 0.15)")):
        for key in keys:
            histories = [json.loads((p/"history.json").read_text()) for p,_ in groups[key]]
            steps = [r["step"] for r in histories[0]]
            values = np.array([[r[metric] for r in hist] for hist in histories])
            mean,std = values.mean(0),values.std(0, ddof=1 if len(values)>1 else 0)
            ax.plot(steps, mean, label=label(key), color=colors.get(key))
            ax.fill_between(steps,mean-std,mean+std,alpha=0.15,color=colors.get(key))
        ax.set_title(title); ax.set_xlabel("Generator updates"); ax.grid(alpha=0.2)
        if metric != "off_manifold": ax.set_yscale("symlog", linthresh=0.001)
    axes[0,0].legend(fontsize=8)
    if len(baseline_by_seed) == 1:
        fig.suptitle(f"Independent validation samples; seed {next(iter(baseline_by_seed))}")
    else:
        fig.suptitle("Independent validation samples; mean +/- sample SD across seeds")
    fig.tight_layout(); fig.savefig(root/"convergence.png",dpi=170); plt.close(fig)
    # Evolution uses the same fixed latent queries within each run.
    first_path = groups[keys[0]][0][0]
    history = json.loads((first_path/"history.json").read_text())
    indices = np.unique(np.linspace(0, len(history)-1, 5).astype(int))
    chosen_steps = [history[i]["step"] for i in indices]
    fig,axes=plt.subplots(len(keys),len(chosen_steps),figsize=(3*len(chosen_steps),2.5*len(keys)),squeeze=False)
    for row,key in enumerate(keys):
        samples=np.load(groups[key][0][0]/"samples.npz")
        for col,step in enumerate(chosen_steps):
            ax=axes[row,col]; pts=samples[f"step_{step}"]
            ax.scatter(*pts.T,s=2,alpha=0.4,color=colors.get(key))
            ax.set_xlim(-2,2);ax.set_ylim(-1.5,1.5);ax.set_aspect("equal")
            if row==0:ax.set_title(f"Step {step}")
            if col==0:ax.set_ylabel(label(key))
    fig.tight_layout();fig.savefig(root/"evolution.png",dpi=150);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    grid=np.linspace(0,0.25,251)
    baseline_samples=[]
    for path,_ in groups[keys[0]]:
        with np.load(path/"samples.npz") as samples:
            baseline_samples.append(samples["baseline"])
    real_dist,_=centerline_distances(np.concatenate(baseline_samples))
    axes[0].plot(grid,[(real_dist<=x).mean() for x in grid],"k--",label="Real samples",linewidth=2)
    for key in keys:
        histograms=[]
        for path,_ in groups[key]:
            with np.load(path/"samples.npz") as samples:
                dist,_=centerline_distances(samples["final_fake"])
            histograms.append([(dist<=x).mean() for x in grid])
        axes[0].plot(grid,np.mean(histograms,axis=0),label=label(key),color=colors.get(key))
    axes[0].axvline(0.05,color="grey",alpha=0.4)
    axes[0].axvline(0.15,color="grey",alpha=0.4)
    axes[0].set(xlabel="Distance to nearest noiseless moon",ylabel="Fraction within distance",
                title="Local thickness and residual off-manifold mass",ylim=(0,1.02))
    axes[0].legend(fontsize=8);axes[0].grid(alpha=0.2)
    for i,(key,row) in enumerate(zip(keys,summary)):
        axes[1].bar(i,row["upper_mass_mean"],yerr=row["upper_mass_std"],color=colors.get(key),capsize=4)
    axes[1].axhline(0.5,color="black",linestyle="--")
    axes[1].set_xticks(range(len(keys)),[label(key) for key in keys],rotation=15)
    axes[1].set(title="Mode balance",ylabel="Mass assigned to upper moon",ylim=(0,1))
    fig.tight_layout();fig.savefig(root/"geometry.png",dpi=170);plt.close(fig)
    print(json.dumps(dict(runs=summary, baseline=baseline_summary),indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--aggregate", type=Path)
    parser.add_argument("--aggregate-seed", type=int)
    parser.add_argument("--method", choices=("energy","sigreg"), default="energy")
    parser.add_argument("--rho", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=8000)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--references", type=int, default=1,
                        help="Independent three-body events averaged per projectile (1 = original TBSM)")
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--directions", type=int, default=32)
    parser.add_argument("--bandwidth", type=float, default=0.3)
    parser.add_argument("--gen-lr", type=float, default=2e-4)
    parser.add_argument("--tracker-lr", type=float, default=1e-3)
    parser.add_argument("--lr-decay", choices=("constant", "cosine"), default="constant")
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--eval-n", type=int, default=1024)
    parser.add_argument("--final-n", type=int, default=4096)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--output", type=Path, default=Path("outputs/two_moons_tbsm"))
    args = parser.parse_args()
    if args.check:
        numerical_checks()
    elif args.aggregate:
        aggregate(args.aggregate, args.aggregate_seed)
    else:
        if not 0 <= args.rho <= 1 or args.bandwidth <= 0:
            parser.error("rho must be in [0,1] and bandwidth must be positive")
        if min(args.references,args.directions,args.batch,args.steps,args.eval_every,args.threads) < 1:
            parser.error("sample counts, steps, evaluation interval, and thread count must be positive")
        if min(args.eval_n,args.final_n) < 2:
            parser.error("unbiased metrics require at least two evaluation samples")
        train(args)
