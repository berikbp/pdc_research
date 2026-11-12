"""Download, verify and benchmark additional models strictly sequentially."""
import argparse
from datetime import datetime,timezone
from pathlib import Path
import subprocess
import sys

from pdc_research.checkpoints import atomic_json,run_lock
from pdc_research.common import read_json,sha256


def run_batch(config_path, output, *, execute=subprocess.run):
    config=read_json(config_path)
    output=Path(output)
    output.mkdir(parents=True,exist_ok=True)
    records=[{'id':m['id'],'output':m['output'],'phase':'pending'} for m in config['models']]
    def progress(phase,**details):
        atomic_json(output/'status.json',{'phase':phase,'batch_id':config['batch_id'],
            'config_sha256':sha256(config_path),'updated_utc':datetime.now(timezone.utc).isoformat(),
            'models':records,**details})
        print(f'Batch phase: {phase} {details}',flush=True)
    with run_lock(output):
        for model,record in zip(config['models'],records):
            try:
                progress('downloading-model',current_model=model['id'])
                execute([sys.executable,'scripts/download_assets.py','--config',model['assets'],'--only','model'],check=True)
                record['phase']='benchmark-pipeline'
                progress('benchmark-pipeline',current_model=model['id'])
                execute([sys.executable,'scripts/run_model_extension.py',
                    '--machine',model['machine'],'--experiment',config['experiment'],
                    '--manifest',config['manifest'],'--output',model['output'],
                    '--pipeline',model['pipeline']],check=True)
                pipeline=read_json(Path(model['pipeline'])/'status.json')
                record['phase']=pipeline['phase']
                if pipeline['phase']!='complete':
                    progress('paused',current_model=model['id'])
                    return
                record['report']=pipeline['report']
            except (OSError,ValueError,RuntimeError,subprocess.CalledProcessError) as exc:
                record.update(phase='failed',error=str(exc))
                # Keep the failed model's evidence, then try the next independent model.
                progress('model-failed',current_model=model['id'])
        if all(r['phase']=='complete' for r in records):
            progress('complete')
        else:
            progress('complete-with-failures')
            raise RuntimeError('One or more models failed; inspect batch status and retained logs')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default='configs/model-batch.json',type=Path)
    parser.add_argument('--output',default='runs/additional-models-batch-v1',type=Path)
    args=parser.parse_args()
    run_batch(args.config,args.output)


if __name__=='__main__':
    main()
