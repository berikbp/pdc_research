"""Prompt-first inference with durable per-trial results and explicit resume."""

import platform
import random
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from .backend import binary, completion, server
from .checkpoints import Checkpoints, atomic_json, run_lock, trial_id, trial_identity
from .common import capture, load_rows, read_json, request, sha256
from .config import configurations
from .workloads import make_cases


class _StopRequested(Exception):
    pass


def _fingerprint(machine: dict, experiment: dict, manifest, executable: str) -> dict:
    dependencies = {
        str(path.resolve()): sha256(path)
        for path in sorted(set(path.resolve() for path in Path(executable).parent.glob("*.so*")))
        if path.is_file()
    }
    return {
        "schema_version": 2, "execution_order": "prompt-first",
        "machine": machine, "experiment": experiment,
        "manifest_sha256": sha256(manifest), "model_sha256": sha256(machine["model"]),
        "server_sha256": sha256(executable), "shared_libraries_sha256": dependencies,
    }


def _initialize(root: Path, identity: dict, rows: list[dict], executable: str, resume: bool) -> None:
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


def _ensure_cases(root: Path, machine: dict, experiment: dict, rows: list[dict]) -> list[dict]:
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


def _check_stop(root: Path) -> None:
    if (root / "STOP").exists():
        raise _StopRequested


def _configuration_trials(checkpoints: Checkpoints, ordinal: int, machine: dict,
                          experiment: dict, threads: int, layers: int) -> None:
    pending = [task for task in checkpoints.pending(ordinal)
               if task["threads"] == threads and task["gpu_layers"] == layers]
    if not pending:
        return
    _check_stop(checkpoints.root)
    cases = {trial_id(trial_identity(case, threads, layers, repetition)): case
             for case in checkpoints.cases
             for repetition in range(experiment["repetitions"])}
    random.Random(experiment["seed"] + ordinal + threads + layers).shuffle(pending)
    log = checkpoints.directory(ordinal) / "logs" / f"t{threads}-g{layers}-{uuid.uuid4().hex}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    with server(machine, threads, layers, log) as base:
        startup_ms = (time.perf_counter() - started) * 1000
        timeout = machine["request_timeout_seconds"]
        warmup_case = cases[trial_id(pending[0])]
        for _ in range(experiment["warmups"]):
            _check_stop(checkpoints.root)
            completion(base, warmup_case["tokens"], experiment, timeout)
        for identity in pending:
            _check_stop(checkpoints.root)
            case = cases[trial_id(identity)]
            record = {**identity, "prompt": ordinal, "server_startup_ms": startup_ms,
                      "server_log": str(log.relative_to(checkpoints.root))}
            try:
                response, elapsed = completion(base, case["tokens"], experiment, timeout)
            except (OSError, ValueError, KeyError, RuntimeError) as exc:
                checkpoints.save(ordinal, {**record, "status": "error", "scope": "trial", "error": str(exc)})
                checkpoints.progress("running", ordinal)
                raise
            checkpoints.save(ordinal, {
                **record, "status": "ok", "client_elapsed_ms": elapsed, "response": response,
            })
            status = checkpoints.progress("running", ordinal)
            print(f"Prompt {ordinal}: {len(checkpoints.expected[ordinal]) - len(checkpoints.pending(ordinal))}"
                  f"/{len(checkpoints.expected[ordinal])}; total {status['successful']}/{status['expected']}",
                  flush=True)


def _drive(checkpoints: Checkpoints, machine: dict, experiment: dict,
           stop_after_prompts: int | None) -> None:
    ordinal = None
    try:
        for ordinal in range(1, len(checkpoints.groups) + 1):
            if not checkpoints.pending(ordinal):
                checkpoints.progress("running", ordinal)
                continue
            if stop_after_prompts is not None and ordinal > stop_after_prompts:
                raise _StopRequested
            _check_stop(checkpoints.root)
            for threads, layers in configurations(machine, experiment["seed"] + ordinal):
                try:
                    _configuration_trials(checkpoints, ordinal, machine, experiment, threads, layers)
                except (OSError, ValueError, KeyError, RuntimeError) as exc:
                    checkpoints.configuration_error(ordinal, {"threads": threads, "gpu_layers": layers}, exc)
                    checkpoints.progress("running", ordinal, str(exc))
                    print(f"Prompt {ordinal}, t{threads}/g{layers} failed: {exc}", flush=True)
            if checkpoints.pending(ordinal):
                raise RuntimeError(f"Prompt {ordinal} incomplete; fix failures and resume this run")
            checkpoints.progress("running", ordinal)
        checkpoints.progress("complete", ordinal)
    except _StopRequested:
        checkpoints.progress("paused", ordinal)
        print("Progress saved. Resume with --resume and the same output directory.", flush=True)
    except KeyboardInterrupt:
        checkpoints.progress("paused", ordinal, "Interrupted; unfinished requests will be retried")
        raise
    except (OSError, ValueError, KeyError, RuntimeError) as exc:
        checkpoints.progress("failed", ordinal, str(exc))
        raise


def run(machine: dict, experiment: dict, manifest: str | Path, output: str | Path,
        *, resume: bool = False, stop_after_prompts: int | None = None) -> None:
    if any(experiment[field] < 1 for field in ("repetitions", "warmups", "output_tokens")):
        raise ValueError("repetitions, warmups and output_tokens must be positive")
    if stop_after_prompts is not None and stop_after_prompts < 1:
        raise ValueError("stop_after_prompts must be positive")
    if not Path(machine["model"]).is_file():
        raise FileNotFoundError(f"Download the GGUF model first: {machine['model']}")
    executable = binary(machine, "llama-server")
    rows = load_rows(manifest)
    if not rows:
        raise ValueError("Empty workload manifest")
    configurations(machine, experiment["seed"])
    root = Path(output)
    if root.exists() and not resume:
        raise FileExistsError(f"Run directory exists: {root}; pass --resume to continue")
    if resume and not root.is_dir():
        raise FileNotFoundError(f"No run to resume: {root}")
    identity = _fingerprint(machine, experiment, manifest, executable)
    root.mkdir(parents=True, exist_ok=resume)
    with run_lock(root):
        _initialize(root, identity, rows, executable, resume)
        atomic_json(root / "status.json", {"phase": "preparing", "finished": False})
        try:
            cases = _ensure_cases(root, machine, experiment, rows)
        except KeyboardInterrupt:
            atomic_json(root / "status.json", {"phase": "paused", "finished": False})
            raise
        except (OSError, ValueError, KeyError, RuntimeError) as exc:
            atomic_json(root / "status.json", {"phase": "failed", "finished": False, "message": str(exc)})
            raise
        try:
            checkpoints = Checkpoints(root, cases, machine, experiment)
        except (OSError, ValueError, KeyError) as exc:
            atomic_json(root / "status.json", {"phase": "failed", "finished": False, "message": str(exc)})
            raise
        checkpoints.progress("running")
        _drive(checkpoints, machine, experiment, stop_after_prompts)
