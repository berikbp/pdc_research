# Multilingual CPU–GPU inference

Reproducible FLORES-200 / llama.cpp experiments investigating whether language
changes the best CPU–GPU inference configuration after token length is controlled.

**Current finding:** on the initial laptop and Qwen2.5-7B Q4_K_M, the controlled
64-input / 64-output-token experiment found no stable, practically meaningful
language-dependent configuration preference. This is a restricted negative RQ1
finding, not proof that language never matters. RQ2 has not been evaluated.

Read the **[current full report](docs/current-report.md)** for all three stages,
the CPU-thread detour, corrected GPU-only baseline, tables and limitations.
The original [research proposal](Research_Proposal.md) remains the source of RQ1/RQ2.

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
| `docs/current-report.md` | Complete research history and current answer |
| `docs/results/` | Small report tables and uncertainty summaries |

Other references: [experiment design](docs/experiment-design.md),
[pause/resume](docs/running-and-resuming.md), [development](docs/development.md),
[CPU thread investigation](docs/cpu-thread-sweep.md),
[GPU placement verification](docs/gpu-baseline.md),
[English/Kazakh inputs](docs/prompts-en-kk.md).
