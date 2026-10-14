# Development

Keep the command line interface small and put experiment behavior in its owning module:

| Module | Responsibility |
| --- | --- |
| `cli.py` | Parse commands and dispatch them |
| `config.py` | Validate JSON settings, enumerate configurations, calculate trial budgets |
| `hardware.py` | Hardware and tool diagnostics |
| `workloads.py` | FLORES alignment, sampling, and token-matched case construction |
| `backend.py` | llama.cpp commands, server lifetime, completion requests and measurement checks |
| `baseline.py` | Synthetic llama-bench execution and artifacts |
| `runner.py` | Real-text experiment orchestration and trial recording |
| `common.py` | JSON, hashing, subprocess capture, and HTTP helpers |

The build script separates argument validation, source checkout, CUDA configuration,
and build metadata recording. Changing Python orchestration does not require a native rebuild.

Run from the repository root:

```bash
uv run python -m unittest discover -s tests -v
uv run pdc-research plan --machine configs/machines/berik-wsl.json
```

Tests do not need a model, dataset, or GPU. They cover aligned sampling, token-matched
eligibility, configuration validation, trial budgets, cache/length checks, preserved
failed-trial identities, run-directory protection, and dirty upstream checkout protection.
Validate actual backend changes separately with the smoke experiment in `docs/setup.md`.

Commit source, tests, shared JSON settings, documentation, `.python-version`, and `uv.lock`.
Keep downloaded backends/toolkits, models, prepared data, and measurement artifacts in
`vendor/`, `models/`, `data/`, and `runs/`; these are ignored. Store personal overrides as
`configs/machines/NAME.local.json` or `configs/experiments/NAME.local.json`. Environment
files and credentials are ignored; a sanitized `.env.example` can be committed.

Do not globally ignore CSV, JSONL, figures, or logs by extension: curated research
artifacts may need versioning outside generated directories. Ignoring a file does not
delete it or remove an already tracked file from Git.
