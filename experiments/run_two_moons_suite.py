"""Run the four TBSM arms on one visible GPU, a single seed by default."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("outputs/two_moons_single_reference"))
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--references", type=int, default=1)
    parser.add_argument("--steps", type=int, default=30000)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    script = Path(__file__).with_name("two_moons_tbsm.py")
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    common = dict(steps=args.steps, batch=1024, references=args.references, bandwidth=0.15,
                  lr_decay="cosine", eval_every=3000, device="cuda", final_n=4096,
                  eval_n=1024, width=64, directions=32, gen_lr=2e-4, tracker_lr=1e-3)

    def run(job):
        method, rho, seed = job
        name = f"{method}_rho{rho:g}_seed{seed}"
        run_dir = args.output/name
        if (run_dir/"result.json").exists():
            old = json.loads((run_dir/"config.json").read_text())
            if not all(old[k] == v for k,v in common.items()):
                raise RuntimeError(f"Existing run has incompatible settings: {run_dir}")
            print(f"Already complete: {name}", flush=True)
            return
        command = [sys.executable, str(script), "--output", str(args.output),
                   "--method", method, "--rho", str(rho), "--seed", str(seed)]
        for k,v in common.items():
            command += ["--"+k.replace("_","-"), str(v)]
        print(f"Starting: {name}", flush=True)
        with (args.output/(name+".log")).open("w") as log:
            subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        print(f"Finished: {name}", flush=True)

    jobs = [(method,rho,seed) for seed in args.seeds for method in ("energy","sigreg") for rho in (0.0,0.9)]
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        list(pool.map(run, jobs))
    aggregation = [sys.executable,str(script),"--aggregate",str(args.output)]
    if len(args.seeds) == 1:
        aggregation += ["--aggregate-seed",str(args.seeds[0])]
    subprocess.run(aggregation, env=env,check=True)


if __name__ == "__main__":
    main()
