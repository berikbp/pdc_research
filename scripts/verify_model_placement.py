"""Short graph-placement diagnostics for the declared CPU/GPU embedding baseline.

This script performs unmeasured generations, saves verbose logs, and refuses a
GPU-model-only label if any logged compute node is assigned to CPU. It does not
trace all timed trials and must not be used as a source of latency comparisons.
"""
import argparse
import collections
import os
from pathlib import Path
import re

from pdc_research.backend import completion, server
from pdc_research.common import load_rows, read_json, request, sha256
from pdc_research.checkpoints import atomic_json
from pdc_research.config import load_configs


def placements(path):
    nodes = []
    for line in path.read_text().splitlines():
        match = re.search(r'node #\s*\d+\s*\(\s*([^)]*?)\s*\):.*?\[\s*(\S+)\s+\].*?use=\d+,c=1', line)
        if match:
            nodes.append((match[1].strip(), match[2]))
    return nodes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--machine', required=True)
    parser.add_argument('--experiment', required=True)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    machine, experiment = load_configs(args.machine, args.experiment)
    if args.output.exists():
        raise FileExistsError('Use a new diagnostic output directory')
    args.output.mkdir(parents=True)
    # Same first passage in all four languages; each tokenized by this model.
    rows = load_rows(args.manifest)[:len(experiment['languages'])]
    if {r['language'] for r in rows} != set(experiment['languages']):
        raise ValueError('Expected one diagnostic input in every configured language')
    os.environ['GGML_SCHED_DEBUG'] = '2'
    results = {}
    diagnostic = {**experiment, 'output_tokens': 2}
    for label, buffer in [('cpu-embedding','CPU'), ('gpu-model-only','CUDA0')]:
        runtime = {**machine, 'tensor_overrides': [f'token_embd.weight={buffer}']}
        log = args.output / f'{label}.log'
        with server(runtime, 1, 99, log, extra_args=('-lv','5')) as base:
            for row in rows:
                tokens = request(base, '/tokenize', {'content': row['text'], 'add_special': True,
                                                    'parse_special': False}, 60)['tokens']
                completion(base, tokens, diagnostic, machine['request_timeout_seconds'])
        nodes = placements(log)
        counts = dict(collections.Counter(backend for _,backend in nodes))
        if not nodes or not any(b.startswith('CUDA') for _,b in nodes):
            raise ValueError(f'{label}: no CUDA compute-node assignments found')
        if label == 'gpu-model-only' and any(not b.startswith('CUDA') for _,b in nodes):
            raise ValueError('GPU-only diagnostic contains non-CUDA compute assignments')
        if label == 'cpu-embedding':
            if not any(b == 'CPU' and op == 'GET_ROWS' for op,b in nodes):
                raise ValueError('CPU embedding assignment was not observed')
            if any(b == 'CPU' and op != 'GET_ROWS' for op,b in nodes):
                raise ValueError('Unexpected additional CPU model operations: inspect log')
        results[label] = {'compute_node_assignment_counts': counts,
                          'log_sha256': sha256(log), 'log': str(log), 'verified': True}
        print(f'Verified {label}: {counts}', flush=True)
    atomic_json(args.output/'verification.json', {
        'model_sha256': sha256(machine['model']), 'machine': machine,
        'input_languages': [r['language'] for r in rows], 'results': results,
        'scope': 'Short graph assignment diagnostics, including setup graphs; not timed results or per-trial tracing.'})


if __name__ == '__main__':
    main()
