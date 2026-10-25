"""Validate and analyze the controlled RQ1 run without third-party dependencies."""
import argparse
import collections
import csv
import hashlib
import itertools
import json
import math
from pathlib import Path
import random
import statistics as st


def quantile(values, probability):
    values = sorted(values)
    p = (len(values) - 1) * probability
    lo = math.floor(p)
    return values[lo] + (values[math.ceil(p)] - values[lo]) * (p - lo)


def intervals(vectors, samples=10000, seed=20261025):
    """Resample groups jointly; familywise centered maximum standardized error."""
    keys = list(vectors)
    n = len(vectors[keys[0]])
    means = {k: st.mean(vectors[k]) for k in keys}
    se = {k: st.stdev(vectors[k]) / math.sqrt(n) for k in keys}
    rng = random.Random(seed)
    draws = {k: [] for k in keys}
    maxima = []
    for _ in range(samples):
        counts = collections.Counter(rng.randrange(n) for _ in range(n))
        maximum = 0.0
        for k in keys:
            value = sum(vectors[k][i] * count for i, count in counts.items()) / n
            draws[k].append(value)
            if se[k] > 0:
                maximum = max(maximum, abs(value - means[k]) / se[k])
        maxima.append(maximum)
    critical = quantile(maxima, .95)
    return {k: {'estimate': means[k], 'ci95_low': quantile(draws[k], .025),
                'ci95_high': quantile(draws[k], .975),
                'simultaneous95_low': means[k] - critical * se[k],
                'simultaneous95_high': means[k] + critical * se[k]} for k in keys}


def write_csv(path, rows):
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def analyze(root, output, samples=10000):
    meta = json.loads((root/'metadata.json').read_text())
    experiment = meta['fingerprint']['experiment']
    configs = [c['id'] for c in experiment['configurations']]
    definitions = {c['id']: c for c in experiment['configurations']}
    cases_bytes = (root/'cases.json').read_bytes()
    assert hashlib.sha256(cases_bytes).hexdigest() == meta['cases_sha256'], 'Cases checksum mismatch'
    cases = [c for c in json.loads(cases_bytes) if c['mode'] == 'matched-64']
    groups = sorted({c['group_id'] for c in cases})
    languages = experiment['languages']
    assert len(groups) == 19 and len(cases) == 76
    rounds = list(range(1, experiment['repetitions']+1))
    expected = set(itertools.product(configs, groups, languages, rounds))
    measured = {}
    digest = hashlib.sha256()
    startup = {}
    flat = []
    for path in sorted(root.glob('configurations/*/prompts/*/results/*.json')):
        data = path.read_bytes()
        digest.update(str(path.relative_to(root)).encode()+b'\0'+data+b'\0')
        r = json.loads(data)
        key = r['configuration'], r['group_id'], r['language'], r['round']
        assert key in expected and key not in measured, f'Duplicate/unexpected trial: {path}'
        cfg = definitions[r['configuration']]
        assert path.relative_to(root).parts[1] == cfg['id']
        assert r['threads'] == cfg['threads'] and r['gpu_layers'] == cfg['gpu_layers']
        assert r['tensor_overrides'] == cfg.get('tensor_overrides', [])
        assert r['repetition'] == r['round']-1 and r['mode'] == 'matched-64'
        t = r['response']['timings']
        assert r['status'] == 'ok' and r['input_tokens'] == t['prompt_n'] == t['predicted_n'] == 64
        assert t['cache_n'] == 0 and not r['response'].get('truncated')
        assert t['prompt_ms'] > 0 and t['predicted_ms'] > 0
        measured[key] = (t['prompt_ms']+t['predicted_ms'])/1000
        startup[r['configuration'],r['server_log']] = r['server_startup_ms']/1000
        flat.append({'configuration': key[0], 'group': key[1], 'language': key[2], 'round': key[3],
                     'input_tokens': 64, 'output_tokens': 64, 'prefill_ms': t['prompt_ms'],
                     'decode_ms': t['predicted_ms'], 'total_s': measured[key],
                     'decode_tokens_per_s': t['predicted_per_second'], 'saved_utc': r['saved_utc']})
    assert set(measured) == expected, 'Missing trials'
    assert len(startup) == 27, 'Unexpected block reloads'
    errors = list(root.glob('configurations/*/prompts/*/errors/*.json'))
    assert not errors, 'Error attempts require review'
    summary = []
    for cfg in configs:
        rs = [r for r in flat if r['configuration'] == cfg]
        summary.append({'configuration': cfg, 'n': len(rs),
                        'prefill_ms': st.mean(r['prefill_ms'] for r in rs),
                        'decode_ms': st.mean(r['decode_ms'] for r in rs),
                        'mean_total_s': st.mean(r['total_s'] for r in rs),
                        **{f'round_{rnd}_s': st.mean(r['total_s'] for r in rs if r['round']==rnd)
                           for rnd in rounds}})
    language_summary = [{'configuration': cfg,'language': lang,
                         'mean_total_s': st.mean(measured[cfg,g,lang,r] for g in groups for r in rounds)}
                        for cfg in configs for lang in languages]
    baseline = 'gpu-model-only'
    # Paired ratios within each group/language/round, then average the rounds.
    ratios = {(a,b,lang): [st.mean(100*(measured[a,g,lang,r]/measured[b,g,lang,r]-1)
                                  for r in rounds) for g in groups]
              for a,b in itertools.combinations(configs,2) for lang in languages}
    relative = {(c,lang): [st.mean(100*(measured[c,g,lang,r]/measured[baseline,g,lang,r]-1)
                                 for r in rounds) for g in groups]
                for c in configs if c != baseline for lang in languages}
    contrasts = {(a,b,l1,l2): [x-y for x,y in zip(ratios[a,b,l1],ratios[a,b,l2])]
                 for a,b in itertools.combinations(configs,2)
                 for l1,l2 in itertools.combinations(languages,2)}
    relative_ci = intervals(relative, samples)
    interaction_ci = intervals(contrasts, samples)
    relative_rows = [{'configuration': c,'language': lang,**v} for (c,lang),v in relative_ci.items()]
    interaction_rows = []
    for (a,b,l1,l2),v in interaction_ci.items():
        per_round = [st.mean(100*(measured[a,g,l1,r]/measured[b,g,l1,r]-1)
                              -100*(measured[a,g,l2,r]/measured[b,g,l2,r]-1)
                              for g in groups) for r in rounds]
        interaction_rows.append({'configuration_a':a,'configuration_b':b,'language_1':l1,'language_2':l2,
                                 **v, **{f'round_{r}_pp':per_round[r-1] for r in rounds},
                                 'same_sign_all_rounds':all(x>0 for x in per_round) or all(x<0 for x in per_round)})
    significant = [r for r in interaction_rows if r['simultaneous95_low']>0 or r['simultaneous95_high']<0]
    equivalent = [r for r in interaction_rows if r['simultaneous95_low']>=-2 and r['simultaneous95_high']<=2]
    stable_practical = [r for r in significant if abs(r['estimate'])>=2 and r['same_sign_all_rounds']]
    output.mkdir(parents=True, exist_ok=True)
    for name,table in [('trials',flat),('configurations',summary),('languages',language_summary),
                       ('relative-to-gpu',relative_rows),('language-interactions',interaction_rows)]:
        write_csv(output/f'{name}.csv',table)
    validation = {'validated_trials':len(flat),'passage_clusters':19,'rounds':3,'blocks':27,'errors':0,
                  'records_sha256':digest.hexdigest(),'bootstrap_samples':samples,'bootstrap_seed':20261025,
                  'interaction_family_size':len(interaction_rows),'relative_family_size':len(relative_rows),
                  'interaction_simultaneous_nonzero':len(significant),
                  'interaction_intervals_inside_2pp':len(equivalent),
                  'stable_practical_interactions':len(stable_practical),
                  'max_abs_interaction_pp':max(abs(r['estimate']) for r in interaction_rows),
                  'max_simultaneous_interaction_bound_pp':max(max(abs(r['simultaneous95_low']),abs(r['simultaneous95_high'])) for r in interaction_rows)}
    (output/'validation.json').write_text(json.dumps(validation,indent=2)+'\n')
    conclusion = ('No stable, practically meaningful language-dependent configuration preference was detected '
                  'among the nine tested configurations at 64 input / 64 output tokens. This supports a '
                  'restricted negative RQ1 finding for this session, model and laptop, not universal equivalence. '
                  'The GPU-model-only candidate and full-offload CPU-embedding configurations form a close '
                  'performance group; partial offload is substantially slower. These results do not justify '
                  'a language-aware selector or answer RQ2.')
    lines = ['# Controlled RQ1 results: 64 input / 64 output tokens', '', conclusion, '',
             'All 2,052 expected records validated: 19 passage groups, four languages, nine configurations and three rounds. No error records; 27 measured configuration loads. Every request processed 64 input tokens, generated 64 tokens, reused no prompt cache and reported no truncation.', '',
             '## Configuration timing', '',
             'Server prompt processing + generation, seconds; loading, warmups and HTTP overhead excluded. Each round cell averages 76 inputs. Each overall mean averages 228 requests.', '',
             '| Configuration | Round 1 | Round 2 | Round 3 | Mean |',
             '|---|---:|---:|---:|---:|']
    for s in summary:
        lines.append(f"| {s['configuration']} | {s['round_1_s']:.4f} | {s['round_2_s']:.4f} | {s['round_3_s']:.4f} | {s['mean_total_s']:.4f} |")
    lines += ['', 'GPU model-only explicitly places the embedding on CUDA with thread parameter 1. The g99 configurations leave the embedding on CPU. g28/g24 assign fewer transformer layers to GPU. CPU orchestration/sampling remains allowed. GPU-only model placement was established in a prior short diagnostic, not traced on every timed request.', '',
              '## Language timing', '', '| Configuration | English | Russian | Kazakh | Chinese |', '|---|---:|---:|---:|---:|']
    for cfg in configs:
        values=[next(x['mean_total_s'] for x in language_summary if x['configuration']==cfg and x['language']==lang) for lang in languages]
        lines.append('| '+cfg+' | '+' | '.join(f'{x:.4f}' for x in values)+' |')
    lines += ['', '## Uncertainty and interpretation', '',
              f"{samples:,} group-cluster bootstrap resamples (seed 20261025); all languages, configurations and rounds of a sampled passage stay together. Marginal percentile intervals and separate simultaneous 95% centered maximum-standardized-error intervals are provided for the 32 baseline contrasts and the 216 language-by-configuration contrasts. These are approximate bootstrap intervals with only 19 independent passage clusters, conditional on the three observed rounds.", '',
              'A language interaction is the difference between two languages in the paired percentage latency change A relative to B, in percentage points. It is averaged across rounds within group, then across groups. This is not the raw language latency difference. The predeclared practical band is ±2 percentage points.', '',
              f"Of 216 interactions, {len(significant)} have simultaneous intervals excluding zero; {len(equivalent)} have simultaneous intervals wholly inside ±2 points. {len(stable_practical)} meet all three criteria: point estimate magnitude at least 2 points, simultaneous interval excluding zero, and same sign in all rounds. Largest absolute interaction estimate: {validation['max_abs_interaction_pp']:.3f} points.", '',
              'Consult language-interactions.csv for the complete family, not selected favorable comparisons. An interval spanning zero is not proof of no effect. An interval extending beyond ±2 points does not establish practical equivalence. Relative-to-GPU intervals are in relative-to-gpu.csv; positive means slower than the GPU-model-only candidate.', '',
              'Limitations: configuration blocks remain vulnerable to GPU clocks, temperature and background activity; rounds are shuffled within one session, not independent machine sessions. Bootstrapping passages does not measure uncertainty across future hardware sessions. A configuration-wide drift can make all languages slower without producing a language interaction. CPU/GPU telemetry was not collected. Prefix clipping changes semantic content; this is one short context length, one dense model, one laptop and a curated dev sample. There is no held-out policy evaluation or RQ2 answer. No broad claim that language can never matter is warranted.', '',
              'The analysis was specified before execution in docs/rq1-controlled.md. Reproduce with:', '',
              '```bash', 'uv run python scripts/analyze_controlled.py --run runs/berik-rq1-controlled-v1 --output docs/results/rq1-controlled-64-v1', '```', '']
    (output/'report.md').write_text('\n'.join(lines))
    print(json.dumps(validation,indent=2))
    print(json.dumps(summary,indent=2))
    print('Largest interactions:',json.dumps(sorted(interaction_rows,key=lambda x:abs(x['estimate']),reverse=True)[:3],indent=2))


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(argv)
    analyze(args.run,args.output)
