# Multilingual CPU–GPU inference

Reproducible FLORES-200 / llama.cpp experiments on when CPU+GPU execution of an
LLM outperforms GPU-only, and whether the request language shifts that boundary.

**Current finding (stage 1, one laptop, native Linux):** when the model fits in
VRAM, moving layers to the CPU never beats GPU-only, and a three-term cost model
explains the loss (about 2.4 ms per token for every block on the CPU). A split
pays only when the model does not fit: Qwen2.5-14B with 41 of 49 layers on the
GPU is 3.1 times faster than the CPU alone. Language shifts the best
configuration only in that memory-bound regime under concurrent requests, and
only through token length.

Read the **[current full report](docs/current-report.md)** for the measurements,
figures, the comparison with APEX and the limitations. The first stage, measured
under WSL with the original questions from the
[research proposal](Research_Proposal.md), is the
[October report](docs/report-2026-10-wsl.md).

The earlier [model/thread investigation](docs/model-thread-extension.md) is a
separate characterization and is not part of the current questions.

For the supervisor meeting, use the [presentation package](docs/presentation/README.md):
editable PowerPoint, slide PDF, one-page summary and speaker notes.

## For teammates

1. Follow [setup](docs/setup.md) to install Python tooling, build the pinned
   backend and download the verified assets.
2. Create an ignored `configs/machines/NAME.local.json` profile for your hardware.
3. Prepare the frozen workload and run a small smoke test first.
4. Follow [the controlled RQ1 protocol](docs/rq1-controlled.md) for replication.
   Use a unique output directory and record hardware/power conditions.

Python orchestration uses Python 3.12+ and the standard library. A GPU, model and
llama.cpp build are needed for inference, but **not** for unit tests:

```bash
uv sync --locked
uv run python -m unittest discover -s tests -q
uv run pdc-research --help
```

After the setup steps, the controlled experiment uses:

```bash
uv run pdc-research run-controlled --machine configs/machines/NAME.local.json --experiment configs/experiments/rq1-controlled.json --manifest data/workloads/pilot-v3.jsonl --output runs/NAME-rq1-001
uv run pdc-research status --output runs/NAME-rq1-001
uv run pdc-research analyze-controlled --run runs/NAME-rq1-001 --output runs/NAME-rq1-001/analysis
```

`run-controlled` launches inference. `status` and `plan` do not. Append `--resume`
to continue compatible unfinished runs; create `RUN/STOP` for a graceful pause.
Only run one benchmark on the machine at a time. Keep it awake and plugged in.
The protocol contains NVIDIA-specific tensor placement; other backends require
separately verified placement and cannot be silently treated as equivalent.

## What belongs in Git

**Include:** source, tests, pinned configuration, setup/protocol documentation,
the current report and curated summary/uncertainty tables in `docs/results/`.

**Keep local:** `runs/`, `data/`, `models/`, `vendor/`, environments, personal
profiles, raw outputs, server logs and per-request analysis exports. These are
ignored, preserved locally and not required to clone the project. Back up raw
measurements separately; summary tables cannot replace them for independent audit.
No model or dataset is redistributed in this repository.

## Layout and documentation

| Path | Purpose |
|---|---|
| `configs/toolchain.json`, `configs/assets.json` | Backend revision and asset revisions/checksums |
| `configs/machines/`, `configs/experiments/` | Hardware profiles and experimental settings |
| `configs/workloads/` | Frozen passage IDs, source URLs and titles |
| `src/pdc_research/` | CLI, workloads, execution, checkpoints and analysis |
| `scripts/` | Native build, asset setup and report-export entrypoints |
| `tests/` | Tests with artificial data and mocked inference |
| `docs/current-report.md` | Current questions, measurements and answers |
| `docs/report-2026-10-wsl.md` | First stage under WSL, superseded |
| `docs/results/` | Small report tables and uncertainty summaries |

Other references: [experiment design](docs/experiment-design.md),
[pause/resume](docs/running-and-resuming.md), [development](docs/development.md),
[CPU thread investigation](docs/cpu-thread-sweep.md),
[GPU placement verification](docs/gpu-baseline.md),
[English/Kazakh inputs](docs/prompts-en-kk.md).
