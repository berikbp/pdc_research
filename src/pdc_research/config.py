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


# Options the runner sets itself; a duplicate would silently change the declared configuration.
RESERVED_SERVER_ARGS = frozenset({
    "-m", "--model", "-t", "--threads", "-tb", "--threads-batch", "-ngl", "--gpu-layers",
    "--n-gpu-layers", "-c", "--ctx-size", "-b", "--batch-size", "-ub", "--ubatch-size",
    "--host", "--port", "-ot", "--override-tensor",
})


def validate_server_args(value) -> None:
    if not isinstance(value, list) or any(not isinstance(item, str) or not item.strip() for item in value):
        raise ValueError("server_args must be a list of nonempty strings")
    reserved = RESERVED_SERVER_ARGS.intersection(value)
    if reserved:
        raise ValueError(f"server_args must not repeat runner-controlled options: {sorted(reserved)}")


def validate_machine(machine: dict) -> None:
    _required(machine, ("machine_id", "model", "llama_bin", "threads", "gpu_layers",
                        "context_size", "batch_size", "ubatch_size", "port",
                        "startup_timeout_seconds", "request_timeout_seconds"), "Machine")
    for field in ("machine_id", "model", "llama_bin"):
        if not isinstance(machine[field], str) or not machine[field].strip():
            raise ValueError(f"{field} must be a nonempty string")
    _integer_choices(machine["threads"], "threads")
    _integer_choices(machine["gpu_layers"], "gpu_layers", minimum=0)
    overrides = machine.get("tensor_overrides", [])
    if not isinstance(overrides, list) or any(
        not isinstance(value, str) or "=" not in value
        or not all(part.strip() for part in value.split("=", 1))
        for value in overrides
    ):
        raise ValueError("tensor_overrides must contain tensor-pattern=buffer strings")
    validate_server_args(machine.get("server_args", []))
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
    if "selection_file" in experiment and (
        not isinstance(experiment["selection_file"], str) or not experiment["selection_file"].strip()
    ):
        raise ValueError("selection_file must be a nonempty path string")
    if experiment.get("execution_order", "prompt-first") not in ("prompt-first", "configuration-first"):
        raise ValueError("Invalid execution_order")
    if "configurations" in experiment:
        configs = experiment["configurations"]
        if not isinstance(configs, list) or not configs:
            raise ValueError("configurations must be a nonempty list")
        ids = []
        for cfg in configs:
            if not isinstance(cfg, dict):
                raise ValueError("Each configuration must be an object")
            cid = cfg.get("id")
            if not isinstance(cid, str) or not cid or not cid.replace("-", "").isascii() or not cid.replace("-", "").isalnum():
                raise ValueError("Configuration IDs must be safe nonempty names")
            ids.append(cid)
            _positive_integer(cfg.get("threads"), "configuration threads")
            _positive_integer(cfg.get("gpu_layers"), "configuration gpu_layers", minimum=0)
            validate_server_args(cfg.get("server_args", []))
        if len(ids) != len(set(ids)):
            raise ValueError("Configuration IDs must be unique")
    mode = experiment.get("input_mode", "matched-64")
    if mode != "aligned" and mode not in [f"matched-{n}" for n in experiment["matched_tokens"]]:
        if "input_mode" in experiment:
            raise ValueError("input_mode must be aligned or a configured matched length")
    if "expected_groups" in experiment:
        _positive_integer(experiment["expected_groups"], "expected_groups")
        if experiment["expected_groups"] > experiment["groups"]:
            raise ValueError("expected_groups cannot exceed groups")
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
    if "configurations" in experiment:
        count = len(experiment["configurations"])
        return {
            "machine": machine, "experiment": experiment,
            "configurations": count, "rounds": experiment["repetitions"],
            "configuration_loads": count * experiment["repetitions"],
            "candidate_groups": experiment["groups"],
            "max_measured_requests": count * experiment["groups"] * len(experiment["languages"]) * experiment["repetitions"],
            "expected_measured_requests": (count * experiment["expected_groups"] * len(experiment["languages"]) * experiment["repetitions"]
                                           if "expected_groups" in experiment else None),
            "note": "Use run-controlled. Aligned inputs keep full passages; matched-length budgets depend on all-language eligibility.",
        }
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
