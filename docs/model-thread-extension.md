# New direction: model-dependent CPU-thread characterization

Prepared 7 November 2026. This follows Stage 2 rather than the original
language-by-configuration RQ1. The original proposal, measurements and restricted
negative RQ1 finding remain unchanged historical artifacts.

## Working question

When the GPU runs the transformer layers, does changing the available CPU
inference thread count improve latency, and does that behavior depend on the
model and embedding placement? This does not claim asynchronous CPU–GPU
cooperation or that more threads automatically add useful computing power.
Thread settings are software parameters, not pinned physical-core allocations.

## New model and workload

The selected model is official Qwen2.5-3B-Instruct GGUF Q4_K_M, a smaller same-family
comparison with the previous 7B model. The choice changes model dimensions and
possibly other details; differences must not be attributed to parameter count
alone. Source:
https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF

Pinned revision, size and SHA-256 are in `configs/assets-qwen3b.json`. A single
2,104,932,768-byte file is downloaded and checked; no merge or backend rebuild is
required. Do not reuse the 7B token cache: tokenize again with the new model.

Same 20 curated FLORES passages, four languages, full natural aligned inputs,
64 output tokens, greedy generation, no prompt cache. This preserves the
Stage-2 workload type; it does not control language input lengths or answer the
original RQ1 anew.

## Nine configurations and three rounds

- All transformer layers requested on GPU in every setting.
- Explicit CPU embedding: 1, 2, 4, 6, 8, 10, 12, 16 threads.
- Explicit CUDA embedding and thread setting 1: GPU-only model-graph candidate.
  CPU orchestration, sampling, HTTP server and driver work remain allowed.
- Each round revisits all nine configurations in a reproducibly shuffled order;
  input ordering is shuffled within each block. This improves on the historical
  blocked repetitions without creating independent machine sessions.
- 80 inputs × nine configurations × three rounds = **2,160 measured requests**,
  27 measured configuration loads and excluded warmups, plus tokenization setup.
- Short verbose placement diagnostics check CPU embedding versus all-CUDA graph
  assignments before starting the timed run. Diagnostics are not timing results
  and do not trace every measured request.

Primary outcome: prompt_ms + predicted_ms, excluding startup. Compare the same
input/round across settings. Analyze all eight thread settings against the GPU
reference using passage-cluster bootstrap intervals with family adjustment.
Document drift and avoid declaring a unique optimum from tiny mean differences.
Power, CPU affinity and GPU utilization telemetry are not measured.

## Reproduction

```bash
uv run python scripts/download_assets.py --config configs/assets-qwen3b.json --only model
uv run python scripts/run_model_extension.py --machine configs/machines/berik-qwen3b.json --experiment configs/experiments/model-thread-rounds.json --output runs/berik-qwen3b-threads-v1 --pipeline runs/qwen3b-thread-pipeline-v1
```

The pipeline verifies placement, starts the resumable run, and analyzes the complete
results. It skips analysis if the benchmark pauses. STOP is supported at the run
folder once inference starts. Resume by rerunning the pipeline with the same
settings and implementation. A failed/incomplete diagnostic requires inspection
and a new diagnostic folder; do not relabel failed placement as GPU-only.

For status:

```bash
uv run pdc-research status --output runs/berik-qwen3b-threads-v1
```

Before that run folder exists, inspect the pipeline's status.json and download
log. The initial launch uses systemd only to keep these jobs independent of the
chat terminal; the Python pipeline also works in a normal terminal. No automatic
shutdown is configured.

## Cross-model interpretation

The old 7B data used blocked repetitions and follow-up sessions; this 3B run uses
shuffled rounds and explicit placement. Its results can motivate a comparison,
but a firm model-by-thread claim needs a fresh 7B run under this same protocol and
comparable machine conditions. Such a rerun is not automatically launched here.
No advantage, novelty or model-specific optimum is assumed in advance.

## Additional-model queue

The next batch keeps the same matrix and three-round protocol, with a neutral
experiment ID in `configs/experiments/model-thread-comparison.json`:

| Order | Model / conversion source | Quantization | Measured requests |
|---|---|---|---:|
| 1 | Qwen/Qwen2.5-1.5B-Instruct-GGUF (official Qwen) | Q4_K_M | 2,160 |
| 2 | bartowski/Mistral-7B-Instruct-v0.3-GGUF (community conversion) | Q4_K_M | 2,160 |

Sources: https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF and
https://huggingface.co/bartowski/Mistral-7B-Instruct-v0.3-GGUF . Revisions, exact
byte sizes and hashes are pinned independently in their asset configurations.
The community conversion source is an additional experimental difference; sharing
Q4_K_M does not make model architectures or tokenizers identical.

The batch is strictly serial: download/check model 1, verify its placement, run
and analyze it, then download/check model 2 and repeat. This avoids simultaneous
benchmarks and avoids downloading another model during timed inference. Each
model gets its own tokenization, provenance, diagnostics, results and report.

```bash
uv run python scripts/run_model_batch.py --config configs/model-batch.json --output runs/additional-models-batch-v1
```

The batch summary is `runs/additional-models-batch-v1/status.json`. Per-model
outputs are `runs/berik-qwen15b-threads-v1` and `runs/berik-mistral7b-threads-v1`.
Inspect pipeline/download logs before benchmark status files exist. Timing
variation is retained and the deeper audit is deferred as requested. Invalid
placement, OOM or download failure is preserved as a failure, not converted into
a timing observation. A failed independent model does not prevent trying the
next one; a deliberate STOP/pause does stop the queue before the next model.
If any model fails, the batch ends with a nonzero exit and an explicit
complete-with-failures status for later investigation. No shutdown is scheduled.
