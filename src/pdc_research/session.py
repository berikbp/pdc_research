"""Shared run provenance, tokenization setup, and graceful pause controls."""

import platform
import uuid
from datetime import datetime, timezone
from pathlib import Path

from .backend import server
from .checkpoints import atomic_json
from .common import capture, read_json, request, sha256
from .workloads import make_cases


class StopRequested(Exception):
    pass


def build_fingerprint(machine: dict, experiment: dict, manifest, executable: str) -> dict:
    dependencies = {
        str(path.resolve()): sha256(path)
        for path in sorted(set(path.resolve() for path in Path(executable).parent.glob("*.so*")))
        if path.is_file()
    }
    return {
        "schema_version": 2, "execution_order": experiment.get("execution_order", "prompt-first"),
        "machine": machine, "experiment": experiment,
        "manifest_sha256": sha256(manifest), "model_sha256": sha256(machine["model"]),
        "server_sha256": sha256(executable), "shared_libraries_sha256": dependencies,
    }


def initialize_run(root: Path, identity: dict, rows: list[dict], executable: str, resume: bool) -> None:
    if resume:
        if not (root / "metadata.json").is_file():
            raise ValueError("Run has no metadata; use a new output directory")
        previous = read_json(root / "metadata.json")
        if previous.get("fingerprint") != identity:
            raise ValueError("Resume refused: model, backend, manifest, settings, or run schema changed")
        if not (root / "inputs.json").exists():
            atomic_json(root / "inputs.json", rows)
        (root / "STOP").unlink(missing_ok=True)  # Explicit resume acknowledges an earlier stop request.
        return
    atomic_json(root / "metadata.json", {
        "fingerprint": identity, "started_utc": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(), "server_version": capture([executable, "--version"]),
        "gpu_before": capture(["nvidia-smi"]), "git": capture(["git", "rev-parse", "HEAD"]),
    })
    atomic_json(root / "inputs.json", rows)


def prepare_cases(root: Path, machine: dict, experiment: dict, rows: list[dict]) -> list[dict]:
    metadata = read_json(root / "metadata.json")
    path = root / "cases.json"
    if metadata.get("cases_sha256"):
        if not path.exists() or sha256(path) != metadata["cases_sha256"]:
            raise ValueError("Saved tokenized cases changed or are missing; resume refused")
        return read_json(path)
    # A separate unmeasured setup server tokenizes the shared workload once.
    log = root / f"tokenizer-{uuid.uuid4().hex}.log"
    with server(machine, machine["threads"][0], machine["gpu_layers"][0], log) as base:
        def tokenize(text: str) -> list[int]:
            return request(base, "/tokenize", {
                "content": text, "add_special": True, "parse_special": False,
            }, machine["request_timeout_seconds"])["tokens"]
        cases = make_cases(rows, experiment, tokenize)
    if any(len(case["tokens"]) + experiment["output_tokens"] > machine["context_size"] for case in cases):
        raise ValueError("Context too small; increase context_size or reduce workload length")
    atomic_json(path, cases)
    metadata["cases_sha256"] = sha256(path)
    atomic_json(root / "metadata.json", metadata)
    modes = ["aligned", *[f"matched-{length}" for length in experiment["matched_tokens"]]]
    coverage = {mode: sum(case["mode"] == mode for case in cases) for mode in modes}
    atomic_json(root / "coverage.json", coverage)
    print(f"Eligible inputs by mode: {coverage}", flush=True)
    return cases


def check_stop(root: Path) -> None:
    if (root / "STOP").exists():
        raise StopRequested
