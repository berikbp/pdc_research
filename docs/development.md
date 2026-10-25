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
| `runner.py` | Prompt-first orchestration, warm-ups, pause controls and resume |
| `checkpoints.py` | Atomic result writes, process locks, progress and per-prompt summaries |
| `common.py` | JSON, hashing, subprocess capture, and HTTP helpers |

The build script separates argument validation, source checkout, CUDA configuration,
and build metadata recording. Changing Python orchestration does not require a native rebuild.
`scripts/download_assets.py` uses `configs/assets.json` to resume and verify asset
downloads, merge model shards, and check extracted dataset files against the archive.

Run from the repository root:

```bash
uv run python -m unittest discover -s tests -v
uv run pdc-research plan --machine configs/machines/berik-wsl.json
```

Tests do not need a model, dataset, or GPU. They cover aligned sampling, token-matched
eligibility, configuration validation, trial budgets, cache/length checks, preserved
failed-trial identities, run-directory protection, interrupted-run resume, immutable successful results, fingerprint mismatch rejection, atomic writes, process locking, and dirty upstream checkout protection.
Validate actual backend changes separately with the smoke experiment in `docs/setup.md`.

Commit source, tests, shared JSON settings, documentation, `.python-version`, and `uv.lock`.
Keep downloaded backends/toolkits, models, prepared data, and measurement artifacts in
`vendor/`, `models/`, `data/`, and `runs/`; these are ignored. Store personal overrides as
`configs/machines/NAME.local.json` or `configs/experiments/NAME.local.json`. Environment
files and credentials are ignored; a sanitized `.env.example` can be committed.

Do not globally ignore CSV, JSONL, figures, or logs by extension: curated research
artifacts may need versioning outside generated directories. Ignoring a file does not
delete it or remove an already tracked file from Git.

## Current module boundaries

- `session.py` owns shared fingerprinting, tokenization setup and stop handling.
  Both execution modes call its public helpers instead of importing private
  functions from each other.
- `controlled.py` owns the fixed 64-token, nine-configuration, round-first
  protocol and read-only controlled status reconstruction.
- `analysis.py` owns cluster bootstrap analysis and CSV/report generation.
  `scripts/analyze_controlled.py` is a compatibility wrapper; the public command
  is `pdc-research analyze-controlled`.
- `scripts/summarize_history.py` regenerates compact exploratory summaries from
  the four local historical run directories.

Public commands are `doctor`, `prepare`, `plan`, `run`, `run-controlled`,
`baseline`, `status`, and `analyze-controlled`. `run` supports the historical
prompt/configuration-first modes; explicit per-entry placement matrices require
`run-controlled` and are rejected by `run` to prevent an accidental wrong sweep.

Tests and GitHub Actions use fake assets and mock the native backend. They must
pass with only `src/`, `tests/`, `configs/` and `scripts/`; no local GGUF, source
checkout, CUDA toolkit or dataset should be required. CI runs with `PYTHONPATH=src`.

The 2026-10-25 cleanup changes implementation fingerprints. Historical completed
measurements and embedded metadata are retained unchanged and remain analyzable.
Do not rewrite an old fingerprint to force resume after a code/settings change;
use a new run or the exact earlier implementation for unfinished work.

## Research artifacts for Git

Commit `docs/current-report.md` and the compact tables under `docs/results/`.
Per-request `trials.csv` in report export folders is ignored, as are all raw run
folders. The full local record set supports the report but should be backed up
outside the repository. A later data release needs a separate archive; this repo
contains summaries, not the full research dataset.

Before pushing:

```bash
uv run python -m unittest discover -s tests -q
git diff --check
git status --short
```

Review the changed files, commit the code/docs/configs and push using your normal
Git workflow. Do not force-add ignored measurements or downloaded assets.
