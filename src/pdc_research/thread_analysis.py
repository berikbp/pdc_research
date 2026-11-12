"""Summarize a round-based CPU-thread sweep without assuming equal input lengths."""
import csv
import itertools
from pathlib import Path
import statistics

from .analysis import intervals, write_csv
from .common import read_json, sha256
from .controlled import read_controlled_status, select_cases


def analyze_thread_sweep(root, output):
    root, output = Path(root), Path(output)
    status = read_controlled_status(root)
    if not status['finished']:
        raise ValueError('Thread analysis requires a complete run')
    meta = read_json(root/'metadata.json')
    experiment = meta['fingerprint']['experiment']
    if sha256(root/'cases.json') != meta['cases_sha256']:
        raise ValueError('Case checksum mismatch')
    cases = select_cases(read_json(root/'cases.json'), experiment)
    case_lookup = {(c['group_id'],c['language']): c for c in cases}
    configs = {c['id']:c for c in experiment['configurations']}
    rounds = list(range(1,experiment['repetitions']+1))
    expected = {(cfg,g,l,r) for cfg in configs for g,l in case_lookup for r in rounds}
    timings = {}
    for path in sorted(root.glob('configurations/*/prompts/*/results/*.json')):
        record = read_json(path)
        key = record['configuration'],record['group_id'],record['language'],record['round']
        if key not in expected or key in timings:
            raise ValueError(f'Unexpected trial: {path}')
        timing = record['response']['timings']
        if (record['mode'] != experiment.get('input_mode','matched-64')
                or record['repetition'] != key[3]-1
                or record['input_tokens'] != len(case_lookup[key[1],key[2]]['tokens'])
                or timing['prompt_n'] != record['input_tokens']
                or timing['predicted_n'] != experiment['output_tokens']
                or timing['cache_n'] != 0 or record['response'].get('truncated')
                or timing['prompt_ms'] <= 0 or timing['predicted_ms'] <= 0):
            raise ValueError(f'Invalid measurement: {path}')
        timings[key] = (timing['prompt_ms']+timing['predicted_ms'])/1000
    if set(timings) != expected:
        raise ValueError('Incomplete or unbalanced measurements')
    baseline = 'gpu-model-only'
    if baseline not in configs:
        raise ValueError('Expected explicit GPU-model-only reference')
    groups = sorted({c['group_id'] for c in cases})
    languages = experiment['languages']
    comparisons = {}
    for cfg in configs:
        if cfg == baseline:
            continue
        comparisons[cfg] = [statistics.mean(
            100*(timings[cfg,g,l,r]/timings[baseline,g,l,r]-1)
            for l in languages for r in rounds) for g in groups]
    estimates = intervals(comparisons)
    rows = []
    for cfg, definition in configs.items():
        values = [v for k,v in timings.items() if k[0] == cfg]
        rows.append({'configuration':cfg,'threads':definition['threads'],
                     'embedding':definition.get('tensor_overrides',[]),
                     'n':len(values), 'mean_s':statistics.mean(values),
                     **{f'round_{r}_s':statistics.mean(v for k,v in timings.items() if k[0]==cfg and k[3]==r)
                        for r in rounds}})
    output.mkdir(parents=True,exist_ok=True)
    write_csv(output/'thread-summary.csv',rows)
    write_csv(output/'paired-relative-to-gpu.csv',[
        {'configuration':cfg,**estimate} for cfg,estimate in estimates.items()])
    lines = ['# Model-dependent CPU thread investigation','',
             f"Model file: `{meta['fingerprint']['machine']['model']}`. Model SHA-256: `{meta['fingerprint']['model_sha256']}`.",'',
             f"Validated {len(timings)} unique trials across {len(groups)} groups, {len(languages)} languages, {len(configs)} configurations and {len(rounds)} shuffled rounds.",'',
             'The full natural aligned passages are used; input lengths may differ by language/model. '
             'Output budget is fixed. This is CPU-thread characterization, not the previous token-controlled RQ1 test. '
             'CPU-embedding conditions force embedding weights to CPU; the GPU reference forces them to CUDA. '
             'Transformer layers are requested on GPU in all conditions. These settings do not establish asynchronous CPU–GPU cooperation.','',
             '| Configuration | Threads | Round 1, s | Round 2, s | Round 3, s | Mean, s |',
             '|---|---:|---:|---:|---:|---:|']
    for row in rows:
        lines.append(f"| {row['configuration']} | {row['threads']} | " +
                     ' | '.join(f"{row[k]:.4f}" for k in ['round_1_s','round_2_s','round_3_s','mean_s'])+' |')
    lines += ['', 'Paired percentage changes relative to the GPU reference are averaged within passage over languages and rounds. '
              '10,000 cluster bootstrap samples preserve each passage across conditions. '
              'The eight comparisons have approximate simultaneous 95% intervals in paired-relative-to-gpu.csv. '
              'These intervals are conditional on observed rounds; cross-session thermal/clock uncertainty remains. '
              'Sample minima are not necessarily distinct optima.','',
              'Historical 7B comparisons used blocked repetitions and additional sessions. '
              'Do not attribute small cross-model differences to model size or thread count without rerunning both models '
              'under the same protocol. No energy, physical-core allocation, language-quality, novelty or original RQ2 claim is made.','']
    (output/'report.md').write_text('\n'.join(lines))
    print(f'Validated {len(timings)} measurements. Report: {output/"report.md"}')
