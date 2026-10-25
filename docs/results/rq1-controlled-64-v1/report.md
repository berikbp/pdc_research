# Controlled RQ1 results: 64 input / 64 output tokens

No stable, practically meaningful language-dependent configuration preference was detected among the nine tested configurations at 64 input / 64 output tokens. This supports a restricted negative RQ1 finding for this session, model and laptop, not universal equivalence. The GPU-model-only candidate and full-offload CPU-embedding configurations form a close performance group; partial offload is substantially slower. These results do not justify a language-aware selector or answer RQ2.

All 2,052 expected records validated: 19 passage groups, four languages, nine configurations and three rounds. No error records; 27 measured configuration loads. Every request processed 64 input tokens, generated 64 tokens, reused no prompt cache and reported no truncation.

## Configuration timing

Server prompt processing + generation, seconds; loading, warmups and HTTP overhead excluded. Each round cell averages 76 inputs. Each overall mean averages 228 requests.

| Configuration | Round 1 | Round 2 | Round 3 | Mean |
|---|---:|---:|---:|---:|
| gpu-model-only | 1.4336 | 1.4342 | 1.4312 | 1.4330 |
| g99-t1 | 1.4342 | 1.4349 | 1.4527 | 1.4406 |
| g99-t4 | 1.4353 | 1.4315 | 1.4342 | 1.4337 |
| g99-t8 | 1.4542 | 1.4294 | 1.4343 | 1.4393 |
| g99-t16 | 1.4561 | 1.4331 | 1.4312 | 1.4401 |
| g28-t4 | 1.6792 | 1.6728 | 1.6795 | 1.6772 |
| g28-t8 | 1.6879 | 1.6902 | 1.6887 | 1.6889 |
| g24-t4 | 2.8345 | 2.6039 | 2.7042 | 2.7142 |
| g24-t8 | 2.5049 | 2.4845 | 2.4921 | 2.4938 |

GPU model-only explicitly places the embedding on CUDA with thread parameter 1. The g99 configurations leave the embedding on CPU. g28/g24 assign fewer transformer layers to GPU. CPU orchestration/sampling remains allowed. GPU-only model placement was established in a prior short diagnostic, not traced on every timed request.

## Language timing

| Configuration | English | Russian | Kazakh | Chinese |
|---|---:|---:|---:|---:|
| gpu-model-only | 1.4335 | 1.4326 | 1.4342 | 1.4318 |
| g99-t1 | 1.4419 | 1.4389 | 1.4453 | 1.4364 |
| g99-t4 | 1.4323 | 1.4312 | 1.4369 | 1.4344 |
| g99-t8 | 1.4443 | 1.4379 | 1.4366 | 1.4384 |
| g99-t16 | 1.4400 | 1.4385 | 1.4448 | 1.4371 |
| g28-t4 | 1.6736 | 1.6792 | 1.6761 | 1.6798 |
| g28-t8 | 1.6847 | 1.6907 | 1.6915 | 1.6888 |
| g24-t4 | 2.7176 | 2.7210 | 2.7105 | 2.7077 |
| g24-t8 | 2.4995 | 2.4906 | 2.4867 | 2.4984 |

## Uncertainty and interpretation

10,000 group-cluster bootstrap resamples (seed 20261025); all languages, configurations and rounds of a sampled passage stay together. Marginal percentile intervals and separate simultaneous 95% centered maximum-standardized-error intervals are provided for the 32 baseline contrasts and the 216 language-by-configuration contrasts. These are approximate bootstrap intervals with only 19 independent passage clusters, conditional on the three observed rounds.

A language interaction is the difference between two languages in the paired percentage latency change A relative to B, in percentage points. It is averaged across rounds within group, then across groups. This is not the raw language latency difference. The predeclared practical band is ±2 percentage points.

Of 216 interactions, 1 have simultaneous intervals excluding zero; 201 have simultaneous intervals wholly inside ±2 points. 0 meet all three criteria: point estimate magnitude at least 2 points, simultaneous interval excluding zero, and same sign in all rounds. Largest absolute interaction estimate: 0.846 points.

Consult language-interactions.csv for the complete family, not selected favorable comparisons. An interval spanning zero is not proof of no effect. An interval extending beyond ±2 points does not establish practical equivalence. Relative-to-GPU intervals are in relative-to-gpu.csv; positive means slower than the GPU-model-only candidate.

Limitations: configuration blocks remain vulnerable to GPU clocks, temperature and background activity; rounds are shuffled within one session, not independent machine sessions. Bootstrapping passages does not measure uncertainty across future hardware sessions. A configuration-wide drift can make all languages slower without producing a language interaction. CPU/GPU telemetry was not collected. Prefix clipping changes semantic content; this is one short context length, one dense model, one laptop and a curated dev sample. There is no held-out policy evaluation or RQ2 answer. No broad claim that language can never matter is warranted.

The analysis was specified before execution in docs/rq1-controlled.md. Reproduce with:

```bash
uv run python scripts/analyze_controlled.py --run runs/berik-rq1-controlled-v1 --output docs/results/rq1-controlled-64-v1
```
