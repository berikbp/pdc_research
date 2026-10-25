# GPU-only model baseline: definition and verification

The baseline permits CPU orchestration, sampling, HTTP/server and driver activity,
but aims to keep model graph computation on GPU. It does **not** mean a process
with literally zero CPU threads.

## What the pinned backend does

- `--threads 0` and `--threads-batch 0` become
  `std::thread::hardware_concurrency()` in `common/arg.cpp`; zero is not a way to
  disable CPU workers.
- The minimum setting `-t 1 -tb 1` avoids the additional OpenMP worker parallel
  region in the pinned CPU backend. The calling thread and host work remain.
- `-ngl 99` requests full layer offload; it does not establish GPU placement of
  every graph operation.

## Diagnostic on 18 October 2026

A short real generation with `GGML_SCHED_DEBUG=2` and `-lv 5` showed CPU `GET_ROWS`
embedding operations in the original full-layer-offload configuration. Repeating
with the explicit CUDA embedding override assigned every logged model node to
CUDA0, with no CPU model nodes observed:

```text
-ngl 99 -t 1 -tb 1 -ot token_embd.weight=CUDA0
```

This corrected an earlier assumption that a backend change was necessarily needed.
It also corrected earlier shorthand that called all `-ngl 99` runs “GPU-only.”
The graph dumps include setup/reservation graphs, so node counts are not kernel
execution counts. Placement was checked on a short diagnostic, not traced for
every measured request or every possible input/model. Verify again after changing
the backend, model, tensor layout or hardware backend.

Local evidence remains in `runs/gpu-baseline-verification/`, including placement
logs and `placement-summary.json`. Verbose diagnostic timings were excluded from
performance comparisons. Do not label this as a literal zero-thread benchmark.

## Reproduction

The reviewed machine profile `configs/machines/berik-gpu-model-baseline.json` uses
`tensor_overrides: ["token_embd.weight=CUDA0"]`; the runner passes that as `-ot`.
The controlled RQ1 matrix applies it only to `gpu-model-only`. Other entries retain
their default embedding placement. `CUDA0` is NVIDIA-specific; other backends need
a verified buffer name and new provenance.

Read the [current report](current-report.md) for both the exploratory baseline
and its controlled comparison. Host CPU support is allowed throughout.
