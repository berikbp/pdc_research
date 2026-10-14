"""Orchestrate real-text trials and retain reproducible run artifacts."""

import itertools
import platform
import random
from datetime import datetime, timezone
from pathlib import Path

from .backend import binary, completion, server
from .common import append_row, capture, load_rows, request, sha256, write_json
from .config import configurations
from .workloads import make_cases


def _write_metadata(root: Path, machine: dict, experiment: dict, manifest, executable: str) -> None:
    write_json(root / "metadata.json", {
        "schema_version": 1,
        "machine": machine,
        "experiment": experiment,
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "manifest_sha256": sha256(manifest),
        "model_sha256": sha256(machine["model"]),
        "server_sha256": sha256(executable),
        "server_version": capture([executable, "--version"]),
        "gpu_before": capture(["nvidia-smi"]),
        "git": capture(["git", "rev-parse", "HEAD"]),
    })


def _prepare_cases(base: str, rows: list[dict], machine: dict, experiment: dict, root: Path) -> list[dict]:
    def tokenize(text: str) -> list[int]:
        response = request(base, "/tokenize", {
            "content": text, "add_special": True, "parse_special": False,
        }, machine["request_timeout_seconds"])
        return response["tokens"]

    cases = make_cases(rows, experiment, tokenize)
    if any(len(case["tokens"]) + experiment["output_tokens"] > machine["context_size"] for case in cases):
        raise ValueError("Context too small; increase context_size or reduce workload length")
    modes = ["aligned", *[f"matched-{length}" for length in experiment["matched_tokens"]]]
    coverage = {mode: sum(case["mode"] == mode for case in cases) for mode in modes}
    write_json(root / "cases.json", cases)
    write_json(root / "coverage.json", coverage)
    print(f"Eligible inputs by mode: {coverage}", flush=True)
    return cases


def _run_trials(base: str, cases: list[dict], machine: dict, experiment: dict, config: dict, results) -> None:
    timeout = machine["request_timeout_seconds"]
    for _ in range(experiment["warmups"]):
        completion(base, cases[0]["tokens"], experiment, timeout)
    tasks = list(itertools.product(range(experiment["repetitions"]), cases))
    random.Random(experiment["seed"]).shuffle(tasks)
    for repetition, case in tasks:
        identity = {
            **config,
            "group_id": case["group_id"],
            "language": case["language"],
            "mode": case["mode"],
            "repetition": repetition,
            "input_tokens": len(case["tokens"]),
        }
        try:
            response, elapsed = completion(base, case["tokens"], experiment, timeout)
        except (OSError, ValueError, KeyError, RuntimeError) as exc:
            append_row(results, {**identity, "status": "error", "scope": "trial", "error": str(exc)})
            raise
        append_row(results, {
            **identity, "status": "ok", "client_elapsed_ms": elapsed, "response": response,
        })


def run(machine: dict, experiment: dict, manifest: str | Path, output: str | Path) -> None:
    if any(experiment[field] < 1 for field in ("repetitions", "warmups", "output_tokens")):
        raise ValueError("repetitions, warmups and output_tokens must be positive")
    model = Path(machine["model"])
    if not model.is_file():
        raise FileNotFoundError(f"Download the GGUF model first: {model}")
    executable = binary(machine, "llama-server")
    rows = load_rows(manifest)
    if not rows:
        raise ValueError("Empty workload manifest")
    settings = configurations(machine, experiment["seed"])
    root = Path(output)
    root.mkdir(parents=True, exist_ok=False)
    _write_metadata(root, machine, experiment, manifest, executable)
    cases = None
    failures = 0
    with (root / "trials.jsonl").open("x", encoding="utf-8") as results:
        for threads, layers in settings:
            config = {"threads": threads, "gpu_layers": layers}
            print(f"Running {config}", flush=True)
            log = root / f"server-t{threads}-g{layers}.log"
            try:
                with server(machine, threads, layers, log) as base:
                    if cases is None:
                        cases = _prepare_cases(base, rows, machine, experiment, root)
                    _run_trials(base, cases, machine, experiment, config, results)
            except (OSError, ValueError, KeyError, RuntimeError) as exc:
                failures += 1
                append_row(results, {
                    **config, "status": "error", "scope": "configuration", "error": str(exc),
                })
                print(f"Configuration failed: {exc}", flush=True)
    write_json(root / "status.json", {"failed_configurations": failures, "finished": True})
    if failures:
        raise RuntimeError(f"{failures} configurations failed; inspect {root}")
