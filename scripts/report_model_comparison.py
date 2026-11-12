"""Build a model/thread comparison from completed run reports; optionally wait.

Only completed models are summarized. Waiting never starts/restarts a benchmark
or changes its settings, and incomplete/failed cases stay explicitly labeled.
"""
import argparse
import csv
from datetime import datetime,timezone
from pathlib import Path
import subprocess
import time

from pdc_research.checkpoints import atomic_json
from pdc_research.common import read_json

MODELS=[('Qwen2.5-1.5B','runs/berik-qwen15b-threads-v1'),
        ('Qwen2.5-3B','runs/berik-qwen3b-threads-v1'),
        ('Mistral-7B-Instruct-v0.3','runs/berik-mistral7b-threads-v1')]


def load_model(label,path):
    root=Path(path)
    if not (root/'status.json').exists():
        return {'model':label,'run':path,'phase':'not-yet-started'}
    state=read_json(root/'status.json')
    value={'model':label,'run':path,'phase':state['phase'],
           'successful':state.get('successful',0),'expected':state.get('expected',2160)}
    if not state.get('finished'):
        return value
    files=[root/'analysis/thread-summary.csv',root/'analysis/paired-relative-to-gpu.csv']
    if not all(p.exists() for p in files):
        value['phase']='analysis-pending'
        return value
    with files[0].open() as f:rows=list(csv.DictReader(f))
    with files[1].open() as f:paired=list(csv.DictReader(f))
    if len(rows)!=9 or any(int(r['n'])!=240 for r in rows):
        raise ValueError(f'Unbalanced summary: {path}')
    meta=read_json(root/'metadata.json')
    fingerprint=meta['fingerprint']
    e=fingerprint['experiment']
    if e.get('input_mode')!='aligned' or e['output_tokens']!=64 or e['repetitions']!=3:
        raise ValueError('Comparison requires aligned inputs / 64 outputs / three rounds')
    configs=e['configurations']
    expected_threads=[1,2,4,6,8,10,12,16]
    cpus=[c for c in configs if c['id'].startswith('cpu-emb-t')]
    if sorted(c['threads'] for c in cpus)!=expected_threads or any(
        c['gpu_layers']!=99 or c['tensor_overrides']!=['token_embd.weight=CPU'] for c in cpus):
        raise ValueError('CPU embedding/thread matrix differs')
    gpu=next(c for c in configs if c['id']=='gpu-model-only')
    if gpu['threads']!=1 or gpu['gpu_layers']!=99 or gpu['tensor_overrides']!=['token_embd.weight=CUDA0']:
        raise ValueError('GPU model reference differs')
    value.update(phase='complete',rows=rows,paired=paired,
                 errors=len(list(root.glob('configurations/*/prompts/*/errors/*.json'))),
                 provenance={'model_sha256':fingerprint['model_sha256'],
                             'manifest_sha256':fingerprint['manifest_sha256'],
                             'server_sha256':fingerprint['server_sha256']})
    return value


def build(output):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    models=[load_model(label,path) for label,path in MODELS]
    complete=[m for m in models if m['phase']=='complete']
    if len({m['provenance']['manifest_sha256'] for m in complete})>1 or len({m['provenance']['server_sha256'] for m in complete})>1:
        raise ValueError('Completed models do not share the manifest and backend binary')
    lines=['# Additional-model CPU-thread report','',
           f"Updated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}. Generated from saved per-model reports.",'',
           'The models below run sequentially under the same nine-setting, three-shuffled-round protocol. '
           'Means include prompt processing and 64-token generation; startup and warmups are excluded. '
           'Same source passages do not imply equal token counts across models or languages. '
           'Model, tokenizer, quantization conversion source, and session conditions can all differ; '
           'do not attribute cross-model timing changes solely to model size.','',
           '## Completion status','', '| Model | State | Saved / expected |', '|---|---|---:|']
    for m in models:
        lines.append(f"| {m['model']} | {m['phase']} | {m.get('successful',0)} / {m.get('expected',2160)} |")
    if complete:
        lines += ['', '## Fastest observed means','',
                  '| Model | Lowest mean setting | Mean, s | GPU-only mean, s | Best CPU embedding mean, s |',
                  '|---|---|---:|---:|---:|']
        for m in complete:
            best=min(m['rows'],key=lambda r:float(r['mean_s']))
            cpu=min([r for r in m['rows'] if r['configuration']!='gpu-model-only'],key=lambda r:float(r['mean_s']))
            gpu=next(r for r in m['rows'] if r['configuration']=='gpu-model-only')
            lines.append(f"| {m['model']} | {best['configuration']} | {float(best['mean_s']):.4f} | {float(gpu['mean_s']):.4f} | {float(cpu['mean_s']):.4f} |")
        lines += ['', 'These are sample averages, not proof of distinct or universal optimal settings. '
                  'The prior Qwen2.5-7B experiments used blocked repetitions/follow-up sessions and are '
                  'historical context, not part of this table.','']
    for m in complete:
        lines += [f"## {m['model']}: all three rounds",'',
                  '| Configuration | Round 1, s | Round 2, s | Round 3, s | Mean, s |',
                  '|---|---:|---:|---:|---:|']
        for r in m['rows']:
            lines.append('| '+r['configuration']+' | '+' | '.join(f"{float(r[k]):.4f}" for k in ['round_1_s','round_2_s','round_3_s','mean_s'])+' |')
        lines += ['', 'CPU embedding is explicitly placed on CPU; the GPU reference explicitly places it on CUDA. '
                  'Transformer layers are requested on GPU throughout; CPU orchestration remains allowed.','',
                  '| CPU setting vs GPU reference | Paired change, % | Simultaneous 95% interval, % |',
                  '|---|---:|---|']
        for r in m['paired']:
            lines.append(f"| {r['configuration']} | {float(r['estimate']):+.2f} | [{float(r['simultaneous95_low']):+.2f}, {float(r['simultaneous95_high']):+.2f}] |")
        lines += ['', 'Positive means slower than the GPU reference; negative means faster. '
                  'Paired percentages are averaged per passage and therefore need not equal the ratio of aggregate means. '
                  'Intervals resample the 20 passage groups, preserve languages/rounds, and adjust the eight comparisons. '
                  'They are conditional on observed sessions and do not remove temperature/clock/background-load effects.', '',
                  f"Source: `{m['run']}/analysis/`. Recorded error attempts: {m['errors']}.",'']
    lines += ['## Interpretation limits and deferred work','',
              'Keep all round differences visible. The deeper timing-variation audit is deferred as requested; '
              'neither round drift nor inconsistent winners are silently excluded. Repeated model-specific '
              'behavior can motivate an audit, but it does not by itself prove causality, useful CPU–GPU '
              'cooperation, a novel result or an answer to the original length-controlled RQ1/RQ2. '
              'No additional experiments are started by this report generator.','']
    atomic_json(output/'comparison.json',{'models':models,'all_complete':len(complete)==len(models)})
    # Render atomically so readers never see an unfinished Markdown report.
    temporary=output/'comparison-report.md.part'
    temporary.write_text('\n'.join(lines));temporary.replace(output/'comparison-report.md')
    print(f'Updated {output/"comparison-report.md"}; complete models: {len(complete)}/{len(models)}',flush=True)
    return len(complete)==len(models)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=Path('runs/additional-models-batch-v1'))
    parser.add_argument('--wait-for-unit')
    args=parser.parse_args()
    build(args.output)
    if args.wait_for_unit:
        # Poll the job state only; avoid re-reading measurements during benchmark timing.
        while True:
            state=subprocess.check_output(['systemctl','--user','show',args.wait_for_unit,
                                           '-p','ActiveState','--value'],text=True).strip()
            if state not in ('active','activating','deactivating'):
                break
            time.sleep(10)
        if not build(args.output):
            raise SystemExit('Some models are incomplete; the report lists their actual state.')


if __name__=='__main__':
    main()
