"""Durable result files, process locking, and resumable progress summaries."""

import hashlib
import json
import os
import tempfile
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from .common import read_json


def atomic_json(path: Path, value: dict | list) -> None:
    """Publish a flushed complete file; an interrupted write cannot replace it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, prefix=".checkpoint-", delete=False
        ) as stream:
            temporary = Path(stream.name)
            json.dump(value, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        if os.name == "posix":
            descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(descriptor)
            except OSError:
                pass  # Some mounted filesystems cannot fsync directories.
            finally:
                os.close(descriptor)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


@contextmanager
def run_lock(root: Path):
    """OS locks are released even after a crash; no stale-lock deletion needed."""
    with (root / ".lock").open("a+b") as stream:
        if stream.tell() == 0:
            stream.write(b"0")
            stream.flush()
        stream.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise RuntimeError(f"Another benchmark is already using {root}") from exc
        try:
            yield
        finally:
            if os.name == "nt":
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def trial_identity(case: dict, threads: int, layers: int, repetition: int) -> dict:
    return {
        "group_id": case["group_id"], "language": case["language"], "mode": case["mode"],
        "threads": threads, "gpu_layers": layers, "repetition": repetition,
        "input_tokens": len(case["tokens"]),
    }


def trial_id(identity: dict) -> str:
    return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()


class Checkpoints:
    def __init__(self, root: Path, cases: list[dict], machine: dict, experiment: dict):
        self.root = root
        self.cases = cases
        self.groups = list(dict.fromkeys(case["group_id"] for case in cases))
        self.expected = {}
        for ordinal, group in enumerate(self.groups, 1):
            tasks = [
                trial_identity(case, threads, layers, repetition)
                for case in cases if case["group_id"] == group
                for threads in machine["threads"] for layers in machine["gpu_layers"]
                for repetition in range(experiment["repetitions"])
            ]
            self.expected[ordinal] = {trial_id(task): task for task in tasks}
        self.successes = {}
        for ordinal, tasks in self.expected.items():
            directory = self.directory(ordinal) / "results"
            for path in directory.glob("*.json"):
                record = read_json(path)
                key = path.stem
                if (key not in tasks or record.get("status") != "ok"
                        or any(record.get(field) != value for field, value in tasks[key].items())):
                    raise ValueError(f"Result identity mismatch: {path}")
                self.successes[key] = record

    def directory(self, ordinal: int) -> Path:
        return self.root / "prompts" / f"{ordinal:03d}"

    def pending(self, ordinal: int) -> list[dict]:
        return [task for key, task in self.expected[ordinal].items() if key not in self.successes]

    def save(self, ordinal: int, record: dict) -> None:
        identity = {field: record[field] for field in next(iter(self.expected[ordinal].values()))}
        key = trial_id(identity)
        if key not in self.expected[ordinal]:
            raise ValueError("Attempt does not belong to this experiment")
        record = {**record, "saved_utc": datetime.now(timezone.utc).isoformat(), "trial_id": key}
        if record["status"] == "ok":
            if key in self.successes:
                raise ValueError("Successful result already exists")
            atomic_json(self.directory(ordinal) / "results" / f"{key}.json", record)
            self.successes[key] = record
        else:
            atomic_json(self.directory(ordinal) / "errors" / f"{key}-{uuid.uuid4().hex}.json", record)

    def configuration_error(self, ordinal: int, config: dict, error: Exception) -> None:
        atomic_json(self.directory(ordinal) / "errors" / f"configuration-{uuid.uuid4().hex}.json", {
            **config, "scope": "configuration", "status": "error", "error": str(error),
            "saved_utc": datetime.now(timezone.utc).isoformat(),
        })

    def progress(self, phase: str, ordinal: int | None = None, message: str | None = None,
                 *, persist: bool = True) -> dict:
        summaries = []
        for number, group in enumerate(self.groups, 1):
            expected = self.expected[number]
            complete = sum(key in self.successes for key in expected)
            summaries.append({"prompt": number, "group_id": group, "successful": complete,
                              "expected": len(expected), "complete": complete == len(expected)})
        if phase == "complete" and any(not group["complete"] for group in summaries):
            phase = "incomplete"
        status = {
            "phase": phase, "finished": phase == "complete",
            "successful": len(self.successes), "expected": sum(map(len, self.expected.values())),
            "completed_prompts": sum(group["complete"] for group in summaries),
            "next_prompt": next((group["prompt"] for group in summaries if not group["complete"]), None),
            "prompts": summaries,
        }
        if message:
            status["message"] = message
        if persist and ordinal is not None:
            summary = summaries[ordinal - 1]
            results = [self.successes[key] for key in self.expected[ordinal] if key in self.successes]
            atomic_json(self.directory(ordinal) / "summary.json", {
                **summary,
                "inputs": {case["language"]: case["text"] for case in self.cases
                           if case["group_id"] == self.groups[ordinal - 1]},
                "pending": self.pending(ordinal), "results": results,
                "error_attempts": len(list((self.directory(ordinal) / "errors").glob("*.json"))),
            })
        if persist:
            atomic_json(self.root / "status.json", status)
        return status


def read_status(root: str | Path) -> dict:
    """Count committed results rather than trusting a possibly stale status file."""
    root = Path(root)
    metadata = read_json(root / "metadata.json")
    fingerprint = metadata.get("fingerprint", {})
    if fingerprint.get("schema_version") != 2:
        raise ValueError("Unsupported run format")
    previous = read_json(root / "status.json") if (root / "status.json").exists() else {}
    if not metadata.get("cases_sha256"):
        return previous
    from .common import sha256
    path = root / "cases.json"
    if sha256(path) != metadata["cases_sha256"]:
        raise ValueError("Saved cases checksum mismatch")
    checkpoints = Checkpoints(root, read_json(path), fingerprint["machine"], fingerprint["experiment"])
    return checkpoints.progress(previous.get("phase", "unknown"), persist=False)
