"""Validate shared experiment settings and per-machine configuration."""

import itertools
import random

from .common import read_json


def _positive_integer(value, name: str, minimum: int = 1) -> None:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


def _integer_choices(value, name: str, minimum: int = 1, allow_empty: bool = False) -> None:
    if not isinstance(value, list) or (not value and not allow_empty):
        raise ValueError(f"{name} must be a list" + ("" if allow_empty else " with at least one value"))
    for item in value:
        _positive_integer(item, name, minimum)
    if len(value) != len(set(value)):
        raise ValueError(f"{name} must not contain duplicates")


def _required(config: dict, names: tuple[str, ...], label: str) -> None:
    if not isinstance(config, dict):
        raise ValueError(f"{label} configuration must be a JSON object")
    missing = [name for name in names if name not in config]
    if missing:
        raise ValueError(f"{label} configuration missing: {', '.join(missing)}")


def validate_machine(machine: dict) -> None:
    _required(machine, ("machine_id", "model", "llama_bin", "threads", "gpu_layers",
                        "context_size", "batch_size", "ubatch_size", "port",
                        "startup_timeout_seconds", "request_timeout_seconds"), "Machine")
    for field in ("machine_id", "model", "llama_bin"):
        if not isinstance(machine[field], str) or not machine[field].strip():
            raise ValueError(f"{field} must be a nonempty string")
    _integer_choices(machine["threads"], "threads")
    _integer_choices(machine["gpu_layers"], "gpu_layers", minimum=0)
    for field in ("context_size", "batch_size", "ubatch_size", "port",
                  "startup_timeout_seconds", "request_timeout_seconds"):
        _positive_integer(machine[field], field)
    if machine["port"] > 65535:
        raise ValueError("port must be <= 65535")
    if machine["ubatch_size"] > machine["batch_size"]:
        raise ValueError("ubatch_size must not exceed batch_size")


def validate_experiment(experiment: dict) -> None:
    _required(experiment, ("experiment_id", "split", "languages", "groups",
                           "sentences_per_group", "seed", "repetitions", "warmups",
                           "output_tokens", "matched_tokens", "synthetic_prompt_tokens"), "Experiment")
    if experiment["split"] not in ("dev", "devtest"):
        raise ValueError("split must be dev or devtest")
    languages = experiment["languages"]
    if (not isinstance(languages, list) or not languages
            or any(not isinstance(lang, str) or not lang.replace("_", "").isalnum() for lang in languages)
            or len(languages) != len(set(languages))):
        raise ValueError("languages must be unique nonempty language codes")
    for field in ("groups", "sentences_per_group", "repetitions", "warmups", "output_tokens"):
        _positive_integer(experiment[field], field)
    _positive_integer(experiment["seed"], "seed", minimum=0)
    _integer_choices(experiment["matched_tokens"], "matched_tokens", allow_empty=True)
    _integer_choices(experiment["synthetic_prompt_tokens"], "synthetic_prompt_tokens")


def load_experiment(path: str) -> dict:
    experiment = read_json(path)
    validate_experiment(experiment)
    return experiment


def load_configs(machine_path: str, experiment_path: str) -> tuple[dict, dict]:
    machine = read_json(machine_path)
    validate_machine(machine)
    experiment = load_experiment(experiment_path)
    return machine, experiment


def configurations(machine: dict, seed: int) -> list[tuple[int, int]]:
    values = list(itertools.product(machine["threads"], machine["gpu_layers"]))
    if not values or any(t < 1 or g < 0 for t, g in values):
        raise ValueError("Provide positive thread counts and nonnegative GPU layer counts")
    random.Random(seed).shuffle(values)
    return values


def experiment_plan(machine: dict, experiment: dict) -> dict:
    count = len(configurations(machine, experiment["seed"]))
    inputs = experiment["groups"] * len(experiment["languages"])
    return {
        "machine": machine,
        "experiment": experiment,
        "configurations": count,
        "aligned_inputs": inputs,
        "aligned_measured_requests": count * inputs * experiment["repetitions"],
        "note": "Matched inputs add requests; eligible group counts depend on actual tokenizer lengths.",
    }
