"""Export compact summaries of the exploratory runs; keep raw outputs local."""
import argparse
import collections
import csv
import hashlib
import json
from pathlib import Path
import statistics

RUNS = ['berik-pilot-v3', 'berik-cpu-threads-v1',
        'berik-cpu-threads-10-v1', 'berik-gpu-model-baseline-v1']


def summarize(run_root, output):
    output.mkdir(parents=True, exist_ok=True)
    table, provenance = [], {}
    for name in RUNS:
        root = run_root / name
        metadata = json.loads((root/'metadata.json').read_text())
        groups = collections.defaultdict(list)
        digest = hashlib.sha256()
        count = 0
        for path in sorted(root.glob('prompts/*/results/*.json')):
            data = path.read_bytes()
            digest.update(str(path.relative_to(root)).encode()+b'\0'+data+b'\0')
            r = json.loads(data)
            t = r['response']['timings']
            if (r['status'] != 'ok' or t['predicted_n'] != 64
                    or t['prompt_n'] != r['input_tokens'] or t['cache_n']
                    or r['response'].get('truncated')):
                raise ValueError(f'Invalid measurement: {path}')
            groups[r['threads'], r['gpu_layers']].append(r)
            count += 1
        expected = (960 if name == 'berik-pilot-v3' else
                    1680 if name == 'berik-cpu-threads-v1' else 240)
        if count != expected:
            raise ValueError(f'{name}: expected {expected}, found {count}')
        for (threads, layers), rows in sorted(groups.items()):
            times = [(r['response']['timings']['prompt_ms']+r['response']['timings']['predicted_ms'])/1000
                     for r in rows]
            entry = {'run': name, 'threads': threads, 'gpu_layers': layers, 'n': len(rows),
                     'mean_total_s': statistics.mean(times),
                     'prefill_ms': statistics.mean(r['response']['timings']['prompt_ms'] for r in rows),
                     'decode_tokens_per_s': statistics.mean(r['response']['timings']['predicted_per_second'] for r in rows)}
            for rep in range(3):
                values = [value for value,r in zip(times, rows) if r['repetition']==rep]
                entry[f'repetition_{rep+1}_s'] = statistics.mean(values) if values else ''
            table.append(entry)
        provenance[name] = {'count': count, 'started_utc': metadata['started_utc'],
                            'errors': len(list(root.glob('prompts/*/errors/*.json'))),
                            'records_sha256': digest.hexdigest(),
                            'model_sha256': metadata['fingerprint']['model_sha256'],
                            'server_sha256': metadata['fingerprint']['server_sha256']}
    with (output/'exploratory-configurations.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(table[0]), lineterminator="\n");writer.writeheader();writer.writerows(table)
    (output/'exploratory-provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    print(f'Exported {sum(x["count"] for x in provenance.values())} exploratory measurements to {output}')


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs',type=Path,default=Path('runs'))
    parser.add_argument('--output',type=Path,default=Path('docs/results/history'))
    args=parser.parse_args()
    summarize(args.runs,args.output)
