"""Run one arm on its assigned GPU, optionally adopting an existing 5k block."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import yaml

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'outputs/imagenet_sit_l_siglip256'


def process_identity(pid):
    try:
        # Fields after the final ')' begin at stat field 3 (state).
        fields=Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()
        return fields[0], fields[19]
    except FileNotFoundError:
        return None


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--arm',choices=['energy','sigreg','sigreg_imq'],required=True)
    parser.add_argument('--gpu',type=int,required=True)
    parser.add_argument('--adopt-pid',type=int)
    parser.add_argument('--retire-scheduler',type=int)
    args=parser.parse_args()
    config=ROOT/f'experiments/configs/imagenet_sit_l_siglip256/{args.arm}.yaml'
    cfg=yaml.safe_load(config.read_text())
    out=Path(cfg['experiment']['save_dir'])
    out.mkdir(parents=True,exist_ok=True)
    lock=(out/'runner.lock').open('w')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    status_path=out/'runner_status.json'
    def status(state,**kwargs):
        value=dict(arm=args.arm,physical_gpu=args.gpu,state=state,**kwargs)
        tmp=status_path.with_suffix('.tmp')
        tmp.write_text(json.dumps(value,indent=2)); tmp.replace(status_path)
        print(json.dumps(value),flush=True)
    if args.adopt_pid:
        original=process_identity(args.adopt_pid)
        if original is None:
            raise RuntimeError('Existing energy process disappeared before adoption')
        parent_identity=process_identity(args.retire_scheduler) if args.retire_scheduler else None
        status('adopting_current_block',pid=args.adopt_pid,until=5000)
        while True:
            current=process_identity(args.adopt_pid)
            if current is None or current[0]=='Z' or current[1]!=original[1]:
                break
            time.sleep(10)
        saved=json.loads((out/'status.json').read_text())['step']
        if saved<5000:
            raise RuntimeError(f'Adopted block stopped before checkpoint 5000 (saved {saved}); inspect train.log')
        if args.retire_scheduler and parent_identity:
            current=process_identity(args.retire_scheduler)
            if current and current[1]==parent_identity[1]:
                os.kill(args.retire_scheduler,signal.SIGTERM)
                os.kill(args.retire_scheduler,signal.SIGCONT)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(args.gpu),OMP_NUM_THREADS='8',
             OPENBLAS_NUM_THREADS='8',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
    status('starting',total_steps=cfg['training']['total_steps'])
    command=[sys.executable,'-u',str(ROOT/'experiments/imagenet_tbsm.py'),'train','--config',str(config)]
    with (out/'train.log').open('a') as log:
        child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,cwd=ROOT,env=env)
        status('running',pid=child.pid,total_steps=cfg['training']['total_steps'])
        code=child.wait()
    status('complete' if code==0 else 'failed',exit_code=code)
    if code:
        raise SystemExit(code)


if __name__=='__main__':
    main()
