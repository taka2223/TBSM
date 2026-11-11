"""Paired experiment harness; updates delegate to the original TBSM.step."""
import argparse
import contextlib
import copy
import json
import os
from pathlib import Path
import pickle
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "imgnet"))
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import yaml

from builders import build_training_model
from builders.common import load_config
from builders.moments import ImageNetMomentsDataset, ResumableBatches
from methods.sigreg import SlicedGaussianField, projected_force


def emit(event, **values):
    print(json.dumps({"event": event, **values}), flush=True)


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n")
    tmp.replace(path)


def rng_state():
    return dict(torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state(),
                numpy=np.random.get_state(), python=random.getstate())


def restore_rng(state):
    torch.set_rng_state(state["torch"])
    torch.cuda.set_rng_state(state["cuda"])
    np.random.set_state(state["numpy"])
    random.setstate(state["python"])


@contextlib.contextmanager
def preserve_rng():
    state = rng_state()
    try:
        yield
    finally:
        restore_rng(state)


def autocast(cfg):
    return torch.autocast("cuda", dtype=torch.bfloat16, enabled=cfg["training"]["bf16"])


def setup(cfg):
    seed = cfg["runtime"]["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = False
    assert cfg["method"]["params"]["trainer"]["grad_accum_steps"] == 1
    if cfg["training"]["bf16"] and not torch.cuda.is_bf16_supported():
        raise RuntimeError("BF16 requested but unsupported")
    build_cfg = copy.deepcopy(cfg)
    build_cfg["runtime"].update(device=torch.device("cuda:0"), world_size=1, local_rank=0)
    build_cfg["experiment"]["resume"] = ""
    model, _ = build_training_model(build_cfg)
    emit("initialized", parameters=sum(p.numel() for p in model.generator.parameters()),
         strict_init=cfg["experiment"].get("strict_init"), bf16=cfg["training"]["bf16"])
    return model


def loader(cfg, start=0, batch=None):
    ds = ImageNetMomentsDataset(cfg["data"]["data_dir"])
    bs = batch or cfg["training"]["batch_size"]
    batches = ResumableBatches(len(ds), bs, cfg["runtime"]["seed"], start)
    return iter(DataLoader(ds, batch_sampler=batches, num_workers=cfg["data"]["num_workers"],
                           pin_memory=True, generator=torch.Generator().manual_seed(913),
                           persistent_workers=cfg["data"]["num_workers"] > 0))


def real_batch(cfg, data):
    moments, y = next(data)
    mu, std = moments.cuda(non_blocking=True).chunk(2, dim=1)
    x = (mu + std * torch.randn_like(mu)) * cfg["latent"]["multiplier"]
    return x, y.cuda(non_blocking=True)


def save_checkpoint(model, cfg, step):
    out = Path(cfg["experiment"]["save_dir"])
    tmp = out / "latest.pt.tmp"
    torch.save(dict(model=model.state_dict(), generator_optimizer=model.generator_optimizer.state_dict(),
                    tracker_optimizer=model.tracker_optimizer.state_dict(), step=step,
                    accum_count=model._accum_count, rng=rng_state(), config=cfg), tmp)
    tmp.replace(out / "latest.pt")
    if step % cfg["evaluation"]["every"] == 0:
        # Retain the evaluated EMA even after latest.pt advances.
        torch.save(dict(backbone=model._eval_generator.backbone.state_dict(), step=step),
                   out / f"ema_step_{step:07d}.pt")
    atomic_json(out / "status.json", dict(step=step, checkpoint="latest.pt"))
    emit("checkpoint", step=step, path=str(out / "latest.pt"))


def resume(model, cfg):
    path = Path(cfg["experiment"]["save_dir"]) / "latest.pt"
    if not path.exists():
        return 0
    # Only locally produced complete experiment checkpoints are loaded here.
    state = torch.load(path, map_location="cpu", weights_only=False)
    old_cfg, new_cfg = copy.deepcopy(state["config"]), copy.deepcopy(cfg)
    for candidate in (old_cfg, new_cfg):
        candidate["training"].pop("total_steps", None)
    if old_cfg != new_cfg:
        raise ValueError("Refusing resume with a changed experiment protocol")
    model.load_state_dict(state["model"], strict=True)
    model.generator_optimizer.load_state_dict(state["generator_optimizer"])
    model.tracker_optimizer.load_state_dict(state["tracker_optimizer"])
    model._accum_count = state["accum_count"]
    restore_rng(state["rng"])
    emit("resumed", step=state["step"])
    return state["step"]


@torch.no_grad()
def visualize(model, cfg, step):
    out = Path(cfg["experiment"]["save_dir"]) / "samples"
    out.mkdir(parents=True, exist_ok=True)
    with preserve_rng(), autocast(cfg):
        rng = torch.Generator(device="cuda").manual_seed(10000)
        noise = torch.randn(16, 4, 32, 32, device="cuda", generator=rng)
        y = torch.tensor([42, 281, 517, 867], device="cuda").repeat_interleave(4)
        images = torch.cat([model.sample(y=y[i:i+4], noise=noise[i:i+4]) for i in range(0, 16, 4)])
        save_image(images, out / f"step_{step:07d}.png", nrow=4)
    emit("samples", step=step, path=str(out / f"step_{step:07d}.png"))


@torch.no_grad()
def fid(model, cfg, step, count=None):
    from scipy.linalg import sqrtm
    ecfg = cfg["evaluation"]
    count = ecfg["num_images"] if count is None else count
    sys.path.insert(0, ecfg["pickle_support_path"])
    with preserve_rng():
        with open(ecfg["inception_path"], "rb") as stream:
            detector = pickle.load(stream).eval().requires_grad_(False).cuda()
        # Fixed seed and labels across checkpoints/methods. 10k = 10/class.
        rng = torch.Generator(device="cuda").manual_seed(ecfg["seed"])
        total = torch.zeros(2048, dtype=torch.float64, device="cuda")
        outer = torch.zeros(2048, 2048, dtype=torch.float64, device="cuda")
        for start in range(0, count, ecfg["batch_size"]):
            n = min(ecfg["batch_size"], count - start)
            z = torch.randn(n, 4, 32, 32, device="cuda", generator=rng)
            y = torch.arange(start, start+n, device="cuda") % cfg["data"]["num_classes"]
            with autocast(cfg):
                images = model.sample(y=y, noise=z)
            # Match IMM SDVAE's float-to-uint8 conversion; detector stays FP32.
            pixels = (images * 255).clamp(0, 255).to(torch.uint8)
            feats = detector(pixels, return_features=True).double()
            total.add_(feats.sum(0))
            outer.add_(feats.T @ feats)
            if (start // ecfg["batch_size"]) % 50 == 0:
                emit("fid_progress", step=step, images=start+n, total=count)
        mean = (total / count).cpu().numpy()
        # Population covariance matches the reference FeatureStats convention.
        cov = (outer / count - torch.outer(total / count, total / count)).cpu().numpy()
        del detector, total, outer, feats
        torch.cuda.empty_cache()
        with open(ecfg["ref_stats_path"], "rb") as stream:
            ref = pickle.load(stream)["fid"]
        product_root = sqrtm(cov @ ref["sigma"])
        if np.iscomplexobj(product_root):
            if np.max(np.abs(product_root.imag)) > 1e-3:
                raise FloatingPointError("Non-negligible imaginary part in FID covariance root")
            product_root = product_root.real
        score = float(np.square(mean-ref["mu"]).sum() + np.trace(cov+ref["sigma"]-2*product_root))
        if not np.isfinite(score):
            raise FloatingPointError("Nonfinite FID")
        record = dict(step=step, fid=score, num_images=count, weights="ema", seed=ecfg["seed"],
                      reference=ecfg["ref_stats_path"], covariance="population")
        out = Path(cfg["experiment"]["save_dir"]) / "metrics"
        atomic_json(out / f"fid{count}_step_{step:07d}.json", record)
        emit("fid", **record)
        return record


def finite_logs(logs):
    values = {k: float(v) for k, v in logs.items()}
    if not all(np.isfinite(v) for v in values.values()):
        raise FloatingPointError(f"Nonfinite losses: {values}")
    return values


@torch.no_grad()
def pretrained_sanity(model, cfg):
    """Native SiT integration checks initialization/VAE independently of TBSM."""
    model.generator.eval()
    with preserve_rng(), autocast(cfg):
        rng=torch.Generator(device="cuda").manual_seed(10000)
        x=torch.randn(8,4,32,32,device="cuda",generator=rng)
        y=torch.tensor([42,281,517,867],device="cuda").repeat_interleave(2)
        raw=model.generator.backbone.backbone.model
        for index in range(64):
            t=x.new_full((len(x),),index/64)
            x=x+raw(x,t,y)/64
        images=model.generator.to_image(x)
        out=Path(cfg["experiment"]["save_dir"])
        out.mkdir(parents=True,exist_ok=True)
        save_image(images,out/"pretrained_sit_euler64.png",nrow=2)
        emit("pretrained_sanity",path=str(out/"pretrained_sit_euler64.png"))


def train(cfg, until):
    out = Path(cfg["experiment"]["save_dir"])
    out.mkdir(parents=True, exist_ok=True)
    model = setup(cfg)
    step = resume(model, cfg)
    (out / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    data = loader(cfg, start=step)
    end = min(until or cfg["training"]["total_steps"], cfg["training"]["total_steps"])
    if step == 0:
        visualize(model, cfg, 0)
    # Evaluate a pending boundary after a checkpoint-resume, too.
    if step and step % cfg["evaluation"]["every"] == 0:
        metric = out / "metrics" / f"fid{cfg['evaluation']['num_images']}_step_{step:07d}.json"
        if not metric.exists():
            fid(model, cfg, step)
    with (out / "history.jsonl").open("a", buffering=1) as history:
        while step < end:
            started = time.perf_counter()
            x, y = real_batch(cfg, data)
            with autocast(cfg):
                logs = finite_logs(model.step(x, y))
            step += 1
            torch.cuda.synchronize()
            record = dict(step=step, seconds=time.perf_counter()-started, **logs)
            if step <= 5 or step % cfg["training"]["log_every"] == 0:
                history.write(json.dumps(record) + "\n")
                emit("train", **record)
            if step % cfg["training"]["vis_every"] == 0:
                visualize(model, cfg, step)
            if step % cfg["training"]["save_every"] == 0 or step == end:
                save_checkpoint(model, cfg, step)
            if step % cfg["evaluation"]["every"] == 0:
                fid(model, cfg, step)


@torch.no_grad()
def calibrate(cfg, output):
    # Calibration never modifies a training run: this process is discarded.
    model = setup(cfg)
    data = loader(cfg, batch=8)
    field = model.representation_fields[0]
    sig = SlicedGaussianField(directions=2048, bandwidth=1.0, chunk_size=128, seed=30000).cuda()
    energy_sq, sig_sq, events = 0., 0., []
    model.generator.eval()
    for index in range(4):
        x, y = real_batch(cfg, data)
        t = model._sample_time(x)
        z = torch.randn_like(x)
        with autocast(cfg):
            h = model.generator((1-t[:, None, None, None])*x+t[:, None, None, None]*z, y, t)
            s = model.generator(torch.randn_like(x), y, torch.ones_like(t))
            r, h, s = [field._extract_tokens(model.generator.to_image(v), False) for v in (x,h,s)]
            r, h, s = field._normalize_features(r, h, s)
        _, energy = field._frozen_target(r, h, s, None)
        sliced = sig(h,r,s)
        erms, srms = float(energy.square().mean().sqrt()), float(sliced.square().mean().sqrt())
        energy_sq += erms**2
        sig_sq += srms**2
        events.append(dict(energy_rms=erms, sigreg_rms=srms, t=t.tolist()))
    scale = (energy_sq / sig_sq)**0.5
    report = dict(scale=scale, events=events, seed=cfg["runtime"]["seed"],
                  rationale="One fixed global multiplier matching initial sampled-field RMS; no per-event normalization.")
    atomic_json(output, report)
    emit("calibration", **report)


def probe(cfg, batch, steps, output):
    cfg["training"]["batch_size"] = batch
    result = dict(batch_size=batch, field=cfg["method"]["params"]["field"]["type"],
                  bf16=cfg["training"]["bf16"], grad_accum_steps=1, activation_checkpointing=False)
    try:
        model = setup(cfg)
        data = loader(cfg)
        # Check all gradients just before original AdamW updates, without changing losses.
        grad_reports = []
        def check_grad(optimizer, args, kwargs):
            grads = [p.grad for group in optimizer.param_groups for p in group["params"] if p.grad is not None]
            norms = torch.stack([g.float().norm() for g in grads])
            if not bool(torch.isfinite(norms).all()) or not bool(norms.max() > 0):
                raise FloatingPointError("Invalid or all-zero gradients")
            grad_reports.append(float(norms.norm()))
        model.generator_optimizer.register_step_pre_hook(check_grad)
        model.tracker_optimizer.register_step_pre_hook(check_grad)
        p = next(p for p in model.generator.backbone.parameters() if p.requires_grad)
        before = p.detach().clone()
        torch.cuda.reset_peak_memory_stats()
        records = []
        for index in range(steps):
            started = time.perf_counter()
            x,y = real_batch(cfg,data)
            with autocast(cfg):
                logs = finite_logs(model.step(x,y))
            torch.cuda.synchronize()
            records.append(dict(step=index+1, seconds=time.perf_counter()-started, **logs))
            emit("probe_step", batch_size=batch, **records[-1])
        if torch.equal(p, before):
            raise AssertionError("Generator parameter did not change")
        peak = torch.cuda.max_memory_reserved()
        total = torch.cuda.get_device_properties(0).total_memory
        result.update(status="ok", peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                      peak_reserved_gib=peak/2**30, total_gib=total/2**30,
                      headroom_gib=(total-peak)/2**30, steps=records, gradient_norms=grad_reports)
    except torch.cuda.OutOfMemoryError as exc:
        result.update(status="oom", error=str(exc))
    atomic_json(output, result)
    emit("probe_result", **result)
    return 0 if result["status"] == "ok" else 42


def check():
    torch.manual_seed(11)
    h = torch.randn(5, 7, dtype=torch.float64, requires_grad=True)
    r,s = torch.randn(2,5,7,dtype=torch.float64)
    a = torch.nn.functional.normalize(torch.randn(23,7,dtype=torch.float64),dim=1)
    ell = .7
    # Negative gradient of -k(h,r)+k(h,s) is the desired attraction-minus-repulsion.
    potential = (-torch.exp(-((h-r)@a.T).square()/(2*ell**2))
                 +torch.exp(-((h-s)@a.T).square()/(2*ell**2))).mean(1).sum()
    expected = -torch.autograd.grad(potential,h)[0]
    actual = projected_force(h,r,s,a,ell)/len(a)
    torch.testing.assert_close(expected,actual)
    torch.testing.assert_close(projected_force(h,r,r,a,ell),torch.zeros_like(h))
    pieces = sum(projected_force(h,r,s,chunk,ell) for chunk in a.split(5))/len(a)
    torch.testing.assert_close(actual,pieces)
    f = SlicedGaussianField(directions=23,chunk_size=5)
    state = torch.get_rng_state().clone()
    f(h[:,None,:],r[:,None,:],s[:,None,:])
    assert torch.equal(state,torch.get_rng_state())
    saved = copy.deepcopy(f.state_dict())
    first = f(h[:,None,:],r[:,None,:],s[:,None,:])
    f.load_state_dict(saved)
    torch.testing.assert_close(first,f(h[:,None,:],r[:,None,:],s[:,None,:]),rtol=0,atol=0)
    # Original energy target and both interpolation times.
    from methods.tbsm import RepresentationScatteringField, TBSM
    from networks.universal_adapter import UniversalAdapter
    dummy = torch.nn.Module(); dummy.feat_dim = 7
    field = RepresentationScatteringField(dummy,1.,0.,1000)
    hh,rr,ss = [v.detach()[:,None,:] for v in (h,r,s)]
    target, label = field._frozen_target(rr,hh,ss,None)
    expected = (torch.nn.functional.normalize(rr-hh,dim=-1,eps=1e-6)
                -torch.nn.functional.normalize(ss-hh,dim=-1,eps=1e-6))*7**.5
    torch.testing.assert_close(label,expected,rtol=1e-5,atol=1e-5)
    class Velocity(torch.nn.Module):
        def forward(self,x,y,t):
            self.time=t
            return torch.ones_like(x)*2
    raw=Velocity()
    adapter=UniversalAdapter(raw,source_prediction_type="velocity",source_interpolation_direction="noise_to_data",target_prediction_type="x")
    t=torch.tensor([1.,.5]); x=torch.randn(2,4,2,2)
    torch.testing.assert_close(adapter(x,torch.zeros(2,dtype=torch.long),t),x+2*t[:,None,None,None])
    torch.testing.assert_close(raw.time,1-t)
    sampler=ResumableBatches(31,4,0)
    it=iter(sampler); sequence=[next(it) for _ in range(12)]
    resumed=iter(ResumableBatches(31,4,0,start_batch=9))
    assert [next(resumed) for _ in range(3)]==sequence[9:12]
    emit("checks_passed", checks=["kernel_autograd", "cancellation", "chunk_sum", "private_rng", "field_resume", "energy_target", "sit_time_conversion", "data_resume"])


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("mode",choices=["check","probe","calibrate","train","eval-smoke"])
    parser.add_argument("--config")
    parser.add_argument("--batch",type=int,default=32)
    parser.add_argument("--steps",type=int,default=5)
    parser.add_argument("--until",type=int)
    parser.add_argument("--output")
    args=parser.parse_args()
    if args.mode=="check":
        check(); return
    cfg=load_config(args.config)
    if args.mode=="probe":
        sys.exit(probe(cfg,args.batch,args.steps,args.output))
    elif args.mode=="calibrate":
        calibrate(cfg,args.output)
    elif args.mode=="eval-smoke":
        cfg["experiment"]["save_dir"]=str(ROOT/"outputs/imagenet_sit_l_siglip256/preflight/eval_smoke")
        model=setup(cfg)
        pretrained_sanity(model,cfg)
        visualize(model,cfg,0)
        fid(model,cfg,0,count=32)
    else:
        train(cfg,args.until)


if __name__=="__main__":
    main()
