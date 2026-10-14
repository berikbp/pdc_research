"""Synthetic llama-bench command construction and result recording."""

import subprocess
from pathlib import Path

from .backend import binary
from .common import capture, sha256, write_json


def baseline_command(machine: dict, experiment: dict) -> list[str]:
    return [
        binary(machine, "llama-bench"),
        "-m", machine["model"],
        "-t", ",".join(map(str, machine["threads"])),
        "-ngl", ",".join(map(str, machine["gpu_layers"])),
        "-p", ",".join(map(str, experiment["synthetic_prompt_tokens"])),
        "-n", str(experiment["output_tokens"]),
        "-r", str(experiment["repetitions"]),
        "-b", str(machine["batch_size"]),
        "-ub", str(machine["ubatch_size"]),
        "-fa", "off",
        "-o", "jsonl",
    ]


def run_baseline(machine: dict, experiment: dict, output: str | Path) -> None:
    command = baseline_command(machine, experiment)
    model = Path(machine["model"])
    if not model.is_file():
        raise FileNotFoundError(f"Download the GGUF model first: {model}")
    root = Path(output)
    root.mkdir(parents=True, exist_ok=False)
    write_json(root / "metadata.json", {
        "machine": machine,
        "experiment": experiment,
        "command": command,
        "model_sha256": sha256(model),
        "binary_sha256": sha256(command[0]),
        "version": capture([command[0], "--version"]),
    })
    with (
        (root / "baseline.jsonl").open("x", encoding="utf-8") as stdout,
        (root / "stderr.log").open("x", encoding="utf-8") as stderr,
    ):
        subprocess.run(command, stdout=stdout, stderr=stderr, check=True)
