# PDC research

Reproducible experiments on multilingual tokenization and heterogeneous CPU–GPU
LLM inference using **FLORES-200** and **llama.cpp**.

We ask whether the best CPU–GPU configuration differs across English, Russian,
Kazakh, and Chinese after controlling tokenized sequence length. See the
[research proposal](Research_Proposal.md).

## Start here

Follow [setup and first run](docs/setup.md) to build the pinned backend, obtain
the model/dataset, create your machine profile, and run the pilot.
Read [experiment design](docs/experiment-design.md) before collecting results.

```bash
uv sync --locked
uv run pdc-research doctor
uv run pdc-research plan --machine configs/machines/berik-wsl.json
```

## Repository layout

| Path | Purpose |
| --- | --- |
| `configs/toolchain.json` | Shared llama.cpp revision |
| `configs/machines/` | Hardware-specific paths and execution settings |
| `configs/experiments/` | Shared workload and repetition settings |
| `src/pdc_research/` | CLI, workload preparation, server runner |
| `scripts/setup_llama.py` | Reproducible native backend build |
| `tests/` | Dataset and measurement validation tests |
| `data/`, `models/`, `vendor/`, `runs/` | Generated/downloaded assets, ignored by Git |

## Current status

Implemented: hardware reports, deterministic aligned workload preparation,
synthetic benchmark orchestration, real-text offload sweeps, matched token-prefix
controls, and raw result/provenance recording. The initial profile targets Berik's
Ryzen 7 7435HS / RTX 4060 8 GiB laptop under WSL.

Validated locally: pinned CUDA build and server detection of the RTX 4060 Laptop GPU.
Pending validation: model/dataset downloads and real GPU pilot.
Statistical analysis, telemetry, streaming TTFT, and policy evaluation are later
stages. No performance results have been collected.

The Python runner has no runtime dependencies beyond Python 3.12+. Use new run
directories, retain raw results, and compare model/workload hashes across machines.

For code organization and maintenance checks, see [development notes](docs/development.md).
