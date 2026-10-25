"""Prompt-first and configuration-first inference with durable checkpoints."""

from contextlib import nullcontext

import random
import time
import uuid
from pathlib import Path

from .backend import binary, completion, server
from .checkpoints import Checkpoints, atomic_json, run_lock, trial_id, trial_identity
from .common import load_rows
from .config import configurations
from .session import StopRequested, build_fingerprint, initialize_run, prepare_cases, check_stop


def _configuration_trials(checkpoints: Checkpoints, ordinal: int, machine: dict,
                          experiment: dict, threads: int, layers: int,
                          shared_session: tuple | None = None) -> None:
    pending = [task for task in checkpoints.pending(ordinal)
               if task["threads"] == threads and task["gpu_layers"] == layers]
    if not pending:
        return
    check_stop(checkpoints.root)
    cases = {trial_id(trial_identity(case, threads, layers, repetition)): case
             for case in checkpoints.cases
             for repetition in range(experiment["repetitions"])}
    random.Random(experiment["seed"] + ordinal + threads + layers).shuffle(pending)
    log = checkpoints.directory(ordinal) / "logs" / f"t{threads}-g{layers}-{uuid.uuid4().hex}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    with (nullcontext(shared_session[0]) if shared_session else server(machine, threads, layers, log)) as base:
        startup_ms = shared_session[1] if shared_session else (time.perf_counter() - started) * 1000
        if shared_session:
            log = shared_session[2]
        timeout = machine["request_timeout_seconds"]
        warmup_case = cases[trial_id(pending[0])]
        for _ in range(0 if shared_session else experiment["warmups"]):
            check_stop(checkpoints.root)
            completion(base, warmup_case["tokens"], experiment, timeout)
        for identity in pending:
            check_stop(checkpoints.root)
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
        if experiment.get("execution_order") == "configuration-first":
            for threads, layers in configurations(machine, experiment["seed"]):
                ordinals = [n for n in checkpoints.expected if any(
                    task["threads"] == threads and task["gpu_layers"] == layers
                    for task in checkpoints.pending(n))]
                if not ordinals:
                    continue
                random.Random(experiment["seed"] + threads + layers).shuffle(ordinals)
                check_stop(checkpoints.root)
                log = checkpoints.root / "logs" / f"t{threads}-g{layers}-{uuid.uuid4().hex}.log"
                log.parent.mkdir(parents=True, exist_ok=True)
                started = time.perf_counter()
                with server(machine, threads, layers, log) as base:
                    startup_ms = (time.perf_counter() - started) * 1000
                    for _ in range(experiment["warmups"]):
                        check_stop(checkpoints.root)
                        completion(base, checkpoints.cases[0]["tokens"], experiment,
                                   machine["request_timeout_seconds"])
                    for ordinal in ordinals:
                        _configuration_trials(checkpoints, ordinal, machine, experiment,
                                              threads, layers, (base, startup_ms, log))
            checkpoints.progress("complete", ordinal)
            return
        for ordinal in range(1, len(checkpoints.groups) + 1):
            if not checkpoints.pending(ordinal):
                checkpoints.progress("running", ordinal)
                continue
            if stop_after_prompts is not None and ordinal > stop_after_prompts:
                raise StopRequested
            check_stop(checkpoints.root)
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
    except StopRequested:
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
    if "configurations" in experiment:
        raise ValueError("Explicit configuration matrices require run-controlled")
    if any(experiment[field] < 1 for field in ("repetitions", "warmups", "output_tokens")):
        raise ValueError("repetitions, warmups and output_tokens must be positive")
    if stop_after_prompts is not None and stop_after_prompts < 1:
        raise ValueError("stop_after_prompts must be positive")
    if not Path(machine["model"]).is_file():
        raise FileNotFoundError(f"Download the GGUF model first: {machine['model']}")
    if experiment.get("execution_order") == "configuration-first" and stop_after_prompts is not None:
        raise ValueError("stop-after-prompts requires prompt-first execution; use STOP instead")
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
    identity = build_fingerprint(machine, experiment, manifest, executable)
    root.mkdir(parents=True, exist_ok=resume)
    with run_lock(root):
        initialize_run(root, identity, rows, executable, resume)
        atomic_json(root / "status.json", {"phase": "preparing", "finished": False})
        try:
            cases = prepare_cases(root, machine, experiment, rows)
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
