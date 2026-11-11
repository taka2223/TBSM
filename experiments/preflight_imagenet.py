"""Separate CUDA processes for calibration and descending batch stress tests."""
import json
from pathlib import Path
import subprocess
import sys
import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "experiments/configs/imagenet_sit_l_siglip256"
OUT = ROOT / "outputs/imagenet_sit_l_siglip256/preflight"
ENTRY = ROOT / "experiments/imagenet_tbsm.py"


def run(mode, arm, *args, log):
    command = [sys.executable, str(ENTRY), mode, "--config", str(CONFIGS / f"{arm}.yaml"), *map(str,args)]
    print("Running: " + " ".join(command),flush=True)
    with log.open("w") as stream:
        result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, cwd=ROOT)
    print(log.read_text()[-6000:],flush=True)
    return result.returncode


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    report = OUT / "field_scale.json"
    if not report.exists():
        code = run("calibrate","energy","--output",report,log=OUT / "calibration.log")
        if code:
            raise RuntimeError(f"Calibration failed: {code}")
    cfg = yaml.safe_load((CONFIGS / "sigreg.yaml").read_text())
    cfg["method"]["params"]["field"]["scale"] = json.loads(report.read_text())["scale"]
    (CONFIGS / "sigreg.yaml").write_text(yaml.safe_dump(cfg,sort_keys=False))
    chosen = None
    for batch in [512,256,128,64,32,16,8,4,2,1]:
        passed = True
        for arm in ["energy","sigreg"]:
            result = OUT / f"{arm}_b{batch}.json"
            if not result.exists():
                code = run("probe",arm,"--batch",batch,"--steps",5,"--output",result,
                           log=OUT / f"{arm}_b{batch}.log")
                if code not in [0,42]:
                    raise RuntimeError(f"Probe failed with a non-OOM error: {arm} B{batch} exit={code}")
            value=json.loads(result.read_text())
            if value["status"] != "ok" or value["headroom_gib"] < 5:
                passed = False
                break
        if passed:
            chosen=batch
            break
    if chosen is None:
        raise RuntimeError("No safe common batch size found")
    for arm in ["energy","sigreg"]:
        path=CONFIGS/f"{arm}.yaml"
        cfg=yaml.safe_load(path.read_text())
        cfg["training"]["batch_size"]=chosen
        path.write_text(yaml.safe_dump(cfg,sort_keys=False))
    a,b=[yaml.safe_load((CONFIGS/f"{arm}.yaml").read_text()) for arm in ["energy","sigreg"]]
    for cfg in (a,b):
        cfg["experiment"].pop("save_dir")
        cfg["method"]["params"].pop("field")
    assert a==b,"Unexpected protocol difference"
    (OUT/"selected_batch.json").write_text(json.dumps(dict(batch_size=chosen,minimum_headroom_gib=5),indent=2))
    print(f"Selected common batch={chosen}; configs otherwise identical",flush=True)


if __name__=="__main__":
    main()
