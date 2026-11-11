"""Alternate two resumable single-GPU jobs; never share the GPU concurrently."""
import fcntl
import json
from pathlib import Path
import subprocess
import sys
import time
import yaml

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT / "outputs/imagenet_sit_l_siglip256"
CONFIGS=ROOT / "experiments/configs/imagenet_sit_l_siglip256"


def main():
    BASE.mkdir(parents=True,exist_ok=True)
    lock=(BASE/"runner.lock").open("w")
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    cfgs={arm:yaml.safe_load((CONFIGS/f"{arm}.yaml").read_text()) for arm in ["energy","sigreg"]}
    total=cfgs["energy"]["training"]["total_steps"]
    assert total==cfgs["sigreg"]["training"]["total_steps"]
    assert (BASE/"preflight/selected_batch.json").exists()
    # Start both methods with a short real run before longer alternating blocks.
    endpoints=sorted(set([min(20,total),*range(5000,total+1,5000),total]))
    for end in endpoints:
        for arm,cfg in cfgs.items():
            out=Path(cfg["experiment"]["save_dir"])
            out.mkdir(parents=True,exist_ok=True)
            status=out/"status.json"
            step=json.loads(status.read_text())["step"] if status.exists() else 0
            if step>=end:
                metric=out/"metrics"/f"fid{cfg['evaluation']['num_images']}_step_{step:07d}.json"
                pending_eval=(step==end and step>0 and step%cfg["evaluation"]["every"]==0
                              and not metric.exists())
                if not pending_eval:
                    continue
            current=dict(arm=arm,from_step=step,until=end,total_steps_per_arm=total,
                         started_unix=time.time(),physical_gpu=1)
            (BASE/"runner_status.json").write_text(json.dumps(current,indent=2))
            print(json.dumps(current),flush=True)
            command=[sys.executable,"-u",str(ROOT/"experiments/imagenet_tbsm.py"),"train",
                     "--config",str(CONFIGS/f"{arm}.yaml"),"--until",str(end)]
            with (out/"train.log").open("a") as log:
                code=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,cwd=ROOT).returncode
            if code:
                raise RuntimeError(f"{arm} failed with exit {code}; inspect {out/'train.log'}")
    (BASE/"runner_status.json").write_text(json.dumps(dict(status="complete",steps=total),indent=2))


if __name__=="__main__":
    main()
