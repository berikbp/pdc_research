"""Verify placement, run the new model sweep, and write its report.

Download/check the model first. Optional --wait-for-assets-unit waits for a
separately started systemd download job; systemd is otherwise not required.
"""
import argparse
from datetime import datetime, timezone
from pathlib import Path
import re
import subprocess
import sys
import time

from pdc_research.checkpoints import atomic_json, run_lock
from pdc_research.common import read_json


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--machine',required=True)
    parser.add_argument('--experiment',required=True)
    parser.add_argument('--manifest',default='data/workloads/pilot-v3.jsonl')
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--pipeline',required=True,type=Path)
    parser.add_argument('--wait-for-assets-unit')
    args=parser.parse_args()
    args.pipeline.mkdir(parents=True,exist_ok=True)
    def phase(value,**details):
        atomic_json(args.pipeline/'status.json',{'phase':value,'updated_utc':datetime.now(timezone.utc).isoformat(),**details})
        print(f'Pipeline phase: {value}',flush=True)
    with run_lock(args.pipeline):
        try:
            if args.wait_for_assets_unit:
                unit=args.wait_for_assets_unit
                if not re.fullmatch(r'[a-zA-Z0-9_.-]+',unit):
                    raise ValueError('Invalid systemd unit name')
                phase('downloading-model')
                deadline=time.monotonic()+7200
                while True:
                    state=subprocess.check_output(['systemctl','--user','show',unit,'-p','ActiveState','--value'],text=True).strip()
                    if state not in ('active','activating','deactivating'):
                        break
                    if time.monotonic()>deadline:
                        raise TimeoutError('Model download wait exceeded two hours')
                    time.sleep(5)
                code=subprocess.check_output(['systemctl','--user','show',unit,'-p','ExecMainStatus','--value'],text=True).strip()
                if state=='failed' or code!='0':
                    raise RuntimeError('Model download failed; benchmark was not started')
            common=['--machine',args.machine,'--experiment',args.experiment,'--manifest',args.manifest]
            verification=args.pipeline/'placement'
            if not (verification/'verification.json').exists():
                phase('verifying-model-placement')
                subprocess.run([sys.executable,'scripts/verify_model_placement.py',*common,'--output',str(verification)],check=True)
            benchmark=args.output
            command=[sys.executable,'-m','pdc_research','run-controlled',*common,'--output',str(benchmark)]
            if benchmark.exists():
                command.append('--resume')
            phase('benchmark-running',output=str(benchmark))
            subprocess.run(command,check=True)
            status=read_json(benchmark/'status.json')
            if not status.get('finished'):
                phase('paused',output=str(benchmark),message='Run did not finish; no analysis was attempted')
                return
            phase('analyzing')
            subprocess.run([sys.executable,'-m','pdc_research','analyze-thread-sweep',
                            '--run',str(benchmark),'--output',str(benchmark/'analysis')],check=True)
            phase('complete',output=str(benchmark),report=str(benchmark/'analysis/report.md'))
        except Exception as exc:
            phase('failed',error=str(exc))
            raise


if __name__=='__main__':
    main()
