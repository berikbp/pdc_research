# Prompt-first benchmark: 20 × 4 × 12 = 960 measured requests

The revised pilot uses one repetition, 20 aligned FLORES groups, four languages,
and 12 CPU–GPU configurations. It generates 64 output tokens per measured request.
Matched-token workloads are disabled for this phase. The selection seed remains
42 for reproducibility; it does not mean multiple experimental repetitions.

Review the exact English/Kazakh texts in [the prompt list](prompts-en-kk.md).
Each four-entry passage is confined to one source article, with a coherent topic
and clear opening subject. Original translations are retained and are not guaranteed
to be flawless. Inputs are bare text prefixes,
without a chat template or a question instruction. Generation is greedy continuation,
with EOS ignored to keep the output length fixed. This phase does not assess answer
quality or prove an effect beyond sequence length.

## Start (commands are instructions, not an automatically started experiment)

```bash
uv run pdc-research run --machine configs/machines/berik-wsl.json --output runs/berik-pilot-v3
```

A one-time setup server tokenizes the workload. The runner then processes prompt
1 under all configurations, then prompt 2, and so on. Configuration and language
order are randomized deterministically within each group. It reloads the model
for every group/configuration and performs one unmeasured warm-up after each load.
Therefore 960 is the **measured** count; there are also 240 warm-up generations and
one tokenizer-server setup. Startup time is recorded separately on the trial.
Model loading and tokenization are excluded from measured request latency.

## Pause and resume

To stop automatically after prompt 13:

```bash
uv run pdc-research run --machine configs/machines/berik-wsl.json --output runs/berik-pilot-v3 --stop-after-prompts 13
```

To continue tomorrow, omit the stop limit and add `--resume`:

```bash
uv run pdc-research run --machine configs/machines/berik-wsl.json --output runs/berik-pilot-v3 --resume
```

While running, press Ctrl+C to stop, or create a `STOP` file from another terminal:

```bash
touch runs/berik-pilot-v3/STOP
```

The STOP file allows the current request to finish and be saved before pausing.
Ctrl+C can interrupt the current request; that uncommitted request is retried.
Explicit `--resume` acknowledges and removes an earlier STOP request. The last
in-flight request may need repeating after abrupt shutdown; previously committed
results are not overwritten. Let the runner exit before shutting down when practical.
Atomic writes and fsync reduce partial-write risk, but physical storage failure can
still lose data. Back up the run directory if these measurements are important.

If a configuration fails, its error is saved and other configurations for the
same prompt are attempted. The runner does not advance to the next prompt until
all combinations for the current prompt succeed. Resume retries missing work.
Changing settings/model/backend to fix a failure requires a new run directory;
resume refuses to combine incompatible experiments.
Resume verifies the model and backend hashes before continuing, so the large model
adds some startup time even when earlier prompts are already complete.

## Check results

```bash
uv run pdc-research status --output runs/berik-pilot-v3
```

This counts the committed result files and reports completed groups and the next
group, without loading the model. `phase` is the last saved phase, not a live-process
check. After an abrupt shutdown it may still say `running`; the committed counts
are reconstructed correctly. Status inspection never modifies the run.

| Artifact | Contents |
| --- | --- |
| `metadata.json` | Model, manifest, settings, server and shared-library fingerprints |
| `inputs.json`, `cases.json` | Exact source inputs and tokenizer IDs |
| `coverage.json` | Eligible input counts |
| `status.json` | Saved overall progress |
| `prompts/001/summary.json` | Prompt 1 texts, completed results, pending combinations, errors count |
| `prompts/001/results/*.json` | One immutable successful record per combination, including output and timings |
| `prompts/001/errors/*.json` | Failed attempts retained even after a later successful retry |
| `prompts/001/logs/` | Separate server logs per startup attempt |

Prompt numbers follow the manifest order, not the numerical FLORES sentence IDs.
For example, prompt 1 is `dev-0369`; prompt 13 is `dev-0680`. The pilot default is
`data/workloads/pilot-v3.jsonl`. The earlier mixed-topic manifests are retained as
preparation history; they are no longer the active workload. The frozen passage
list and article URLs are in `configs/workloads/coherent-dev-v1.json`.
Schema-1 run directories cannot be resumed by the new runner.

The initial one-repetition experiment is exploratory. Repeated trials and a
separate matched-length phase are still needed before robust conclusions about RQ1.
