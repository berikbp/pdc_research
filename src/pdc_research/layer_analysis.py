"""Layer-offload sweep analysis: per-placement summaries and a per-block cost model.

Model for decode time per generated token, fitted over configuration means:

    ms_per_token = base + per_cpu_block * k + handoff * [k > 0] + output_on_cpu * [output layer on CPU]

where k is the number of transformer blocks the server actually left on the CPU.
`per_cpu_block` is the extra cost of moving one block from GPU to CPU, not the
absolute cost of a CPU block. Intervals resample passages, keeping all languages,
configurations and rounds of a passage together.

Configurations started with extra server options (for example `--device none`) run
in a different regime; they are summarized but left out of the fit. When no
configuration has every block on the GPU, the handoff term cannot be separated and
`base` is an extrapolated intercept that includes it.
"""

import csv
import json
import random
import statistics as st
from pathlib import Path

from .common import read_json, sha256

TERMS = ("base_ms", "per_cpu_block_ms", "handoff_ms", "output_on_cpu_ms")
BOOTSTRAP_SAMPLES = 2000
BOOTSTRAP_SEED = 20261115


def solve(matrix: list[list[float]], vector: list[float]) -> list[float]:
    """Gaussian elimination with partial pivoting for a small square system."""
    size = len(vector)
    rows = [row[:] + [value] for row, value in zip(matrix, vector)]
    for column in range(size):
        pivot = max(range(column, size), key=lambda r: abs(rows[r][column]))
        if abs(rows[pivot][column]) < 1e-12:
            raise ValueError("Cost model is not identifiable from these configurations")
        rows[column], rows[pivot] = rows[pivot], rows[column]
        for other in range(size):
            if other != column:
                factor = rows[other][column] / rows[column][column]
                rows[other] = [a - factor * b for a, b in zip(rows[other], rows[column])]
    return [rows[i][size] / rows[i][i] for i in range(size)]


def design_row(blocks_on_cpu: int, output_on_gpu: bool) -> list[float]:
    return [1.0, float(blocks_on_cpu), float(blocks_on_cpu > 0), float(not output_on_gpu)]


def fit(points: list[tuple[int, bool, float]]) -> dict:
    """Least squares over (blocks_on_cpu, output_on_gpu, value); unused terms are dropped."""
    design = [design_row(k, on_gpu) for k, on_gpu, _ in points]
    used = [i for i in range(len(TERMS)) if any(row[i] for row in design)
            and (i == 0 or len({row[i] for row in design}) > 1)]
    reduced = [[row[i] for i in used] for row in design]
    values = [value for _, _, value in points]
    normal = [[sum(r[a] * r[b] for r in reduced) for b in range(len(used))] for a in range(len(used))]
    rhs = [sum(r[a] * y for r, y in zip(reduced, values)) for a in range(len(used))]
    solution = solve(normal, rhs)
    coefficients = {TERMS[i]: 0.0 for i in range(len(TERMS))} | {TERMS[i]: v for i, v in zip(used, solution)}
    predicted = [sum(c * x for c, x in zip(solution, row)) for row in reduced]
    mean = st.mean(values)
    total = sum((y - mean) ** 2 for y in values)
    residual = sum((y - p) ** 2 for y, p in zip(values, predicted))
    return {**coefficients, "r_squared": 1 - residual / total if total else 1.0,
            "max_abs_residual": max(abs(y - p) for y, p in zip(values, predicted)),
            "fitted_terms": [TERMS[i] for i in used]}


def quantile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def load_trials(root: Path) -> tuple[dict, list[dict]]:
    """Validate the run against its own declared design and return flat trial rows."""
    metadata = read_json(root / "metadata.json")
    experiment = metadata["fingerprint"]["experiment"]
    if experiment.get("input_mode") != "aligned":
        raise ValueError("Layer analysis expects an aligned-input controlled run")
    if sha256(root / "cases.json") != metadata["cases_sha256"]:
        raise ValueError("Cases checksum mismatch")
    if list(root.glob("configurations/*/prompts/*/errors/*.json")):
        raise ValueError("Run contains error attempts; review them before analysis")
    cases = {(c["group_id"], c["language"]): len(c["tokens"])
             for c in read_json(root / "cases.json") if c["mode"] == "aligned"}
    definitions = {c["id"]: c for c in experiment["configurations"]}
    expected = {(cid, group, language, round_)
                for cid in definitions for group, language in cases
                for round_ in range(1, experiment["repetitions"] + 1)}
    trials, seen = [], set()
    for path in sorted(root.glob("configurations/*/prompts/*/results/*.json")):
        record = read_json(path)
        key = (record["configuration"], record["group_id"], record["language"], record["round"])
        timing, placement = record["response"]["timings"], record.get("placement", {})
        definition = definitions.get(key[0], {})
        if key not in expected or key in seen:
            raise ValueError(f"Unexpected or duplicate trial: {path}")
        if "error" in placement:
            raise ValueError(f"No verified layer placement: {path}: {placement['error']}")
        if (record["status"] != "ok" or timing["prompt_n"] != cases[key[1], key[2]]
                or timing["predicted_n"] != experiment["output_tokens"] or timing.get("cache_n", 0)
                or record["response"].get("truncated")
                or record["gpu_layers"] != definition.get("gpu_layers")
                or placement["offloaded_layers"] != min(record["gpu_layers"], placement["total_layers"])):
            raise ValueError(f"Invalid measurement or placement mismatch: {path}")
        seen.add(key)
        telemetry = record.get("telemetry", {})
        gpu, process = telemetry.get("gpu", {}), telemetry.get("process", {})
        trials.append({
            "configuration": key[0], "group": key[1], "language": key[2], "round": key[3],
            "gpu_layers": record["gpu_layers"], "threads": record["threads"],
            "server_args": " ".join(record.get("server_args", [])),
            "blocks_on_cpu": placement["blocks_on_cpu"], "output_on_gpu": placement["output_layer_on_gpu"],
            "input_tokens": timing["prompt_n"], "output_tokens": timing["predicted_n"],
            "prefill_ms": timing["prompt_ms"], "decode_ms": timing["predicted_ms"],
            "prefill_ms_per_token": timing["prompt_ms"] / timing["prompt_n"],
            "decode_ms_per_token": timing["predicted_ms"] / timing["predicted_n"],
            "total_s": (timing["prompt_ms"] + timing["predicted_ms"]) / 1000,
            "server_vram_mib": (telemetry.get("server_gpu_memory") or {}).get("vram_mib"),
            "gpu_util_pct": gpu.get("gpu_util_mean_pct"), "gpu_power_w": gpu.get("power_mean_w"),
            "gpu_energy_j": gpu.get("energy_j"), "gpu_temp_c": gpu.get("temp_max_c"),
            "cpu_cores": process.get("cpu_cores_mean"), "server_rss_mib": process.get("rss_mib"),
        })
    if seen != expected:
        raise ValueError(f"Incomplete run: {len(seen)} of {len(expected)} trials")
    return experiment, trials


def mean_or_none(values: list) -> float | None:
    present = [value for value in values if value is not None]
    return st.mean(present) if present else None


def summarize(experiment: dict, trials: list[dict]) -> list[dict]:
    rows = []
    for definition in experiment["configurations"]:
        own = [t for t in trials if t["configuration"] == definition["id"]]
        row = {"configuration": definition["id"], "gpu_layers": definition["gpu_layers"],
               "threads": definition["threads"], "server_args": own[0]["server_args"],
               "blocks_on_cpu": own[0]["blocks_on_cpu"], "output_on_gpu": own[0]["output_on_gpu"],
               "n": len(own),
               "decode_ms_per_token": st.mean(t["decode_ms_per_token"] for t in own),
               "decode_tokens_per_s": 1000 / st.mean(t["decode_ms_per_token"] for t in own),
               "prefill_ms_per_token": st.mean(t["prefill_ms_per_token"] for t in own)}
        for language in experiment["languages"]:
            row[f"total_s_{language}"] = st.mean(t["total_s"] for t in own if t["language"] == language)
        for round_ in range(1, experiment["repetitions"] + 1):
            row[f"decode_ms_per_token_round_{round_}"] = st.mean(
                t["decode_ms_per_token"] for t in own if t["round"] == round_)
        for field in ("server_vram_mib", "gpu_util_pct", "gpu_power_w", "cpu_cores", "server_rss_mib",
                      "gpu_temp_c"):
            row[field] = mean_or_none([t[field] for t in own])
        energy = [t["gpu_energy_j"] / (t["input_tokens"] + t["output_tokens"])
                  for t in own if t["gpu_energy_j"] is not None]
        row["gpu_energy_j_per_token"] = st.mean(energy) if energy else None
        rows.append(row)
    return rows


def cost_model(trials: list[dict], metric: str, samples: int = BOOTSTRAP_SAMPLES) -> dict:
    """Fit the block cost model with passage-cluster bootstrap percentile intervals."""
    excluded = sorted({t["configuration"] for t in trials if t["server_args"]})
    trials = [t for t in trials if not t["server_args"]]
    placement = {t["configuration"]: (t["blocks_on_cpu"], t["output_on_gpu"]) for t in trials}
    groups = sorted({t["group"] for t in trials})
    cells: dict[tuple[str, str], list[float]] = {}
    for trial in trials:
        cells.setdefault((trial["configuration"], trial["group"]), []).append(trial[metric])
    per_group = {key: st.mean(values) for key, values in cells.items()}

    def fit_for(selected: list[str]) -> dict:
        return fit([(*placement[cid], st.mean(per_group[cid, group] for group in selected))
                    for cid in placement])

    estimate = fit_for(groups)
    rng = random.Random(BOOTSTRAP_SEED)
    draws = [fit_for([rng.choice(groups) for _ in groups]) for _ in range(samples)]
    return {"metric": metric, **estimate, "excluded_configurations": excluded,
            "full_offload_observed": any(blocks == 0 for blocks, _ in placement.values()),
            "ci95": {term: [quantile([d[term] for d in draws], .025), quantile([d[term] for d in draws], .975)]
                     for term in estimate["fitted_terms"]},
            "bootstrap_samples": samples, "bootstrap_seed": BOOTSTRAP_SEED, "passage_clusters": len(groups)}


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _number(value: float | None, digits: int = 1) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def report(experiment: dict, rows: list[dict], models: dict) -> str:
    decode, prefill = models["decode_ms_per_token"], models["prefill_ms_per_token"]
    languages = experiment["languages"]
    lines = [f"# Layer-offload sweep: {experiment['experiment_id']}", "",
             f"{sum(r['n'] for r in rows):,} validated requests: {len(rows)} configurations, "
             f"{decode['passage_clusters']} passages, {len(languages)} languages, "
             f"{experiment['repetitions']} shuffled rounds, {experiment['output_tokens']} generated tokens. "
             "Full aligned passages are used, so input length differs by language. Placement is the one "
             "the server reported, not the requested `-ngl`.", "",
             "## Cost of moving blocks to the CPU", "",
             "Decode time per generated token, least squares over configuration means:", "",
             "| Term | Estimate, ms | 95% interval |", "|---|---:|---:|"]
    base_name = ("All blocks on GPU" if decode["full_offload_observed"]
                 else "Intercept (extrapolated, includes handoff)")
    names = {"base_ms": base_name, "per_cpu_block_ms": "Each block moved to CPU",
             "handoff_ms": "First CPU block (fixed handoff)", "output_on_cpu_ms": "Output layer on CPU"}
    for term in decode["fitted_terms"]:
        low, high = decode["ci95"][term]
        lines.append(f"| {names[term]} | {decode[term]:.3f} | {low:.3f} to {high:.3f} |")
    lines += ["", f"R² = {decode['r_squared']:.4f}; largest residual {decode['max_abs_residual']:.3f} ms per token. "
              f"Prompt processing: {prefill['per_cpu_block_ms']:.3f} ms per prompt token for each block moved "
              f"to CPU (R² = {prefill['r_squared']:.4f}). Intervals are percentile bootstrap over passages and "
              "do not cover variation between sessions."
              + ("" if decode["full_offload_observed"] else
                 " No configuration holds every block on the GPU, so the intercept is extrapolated, "
                 "not a measured or feasible placement.")
              + ("" if not decode["excluded_configurations"] else
                 " Left out of the fit because extra server options change the regime: "
                 + ", ".join(f"`{name}`" for name in decode["excluded_configurations"]) + "."), "",
              "## Configurations", "",
              "| GPU layers | Blocks on CPU | Decode, tok/s | Decode, ms/tok | Prefill, ms/tok | "
              + " | ".join(f"Total {lang}, s" for lang in languages)
              + " | Server VRAM, MiB | CPU cores | GPU util, % | GPU power, W |",
              "|---:|---:|---:|---:|---:|" + "---:|" * (len(languages) + 4)]
    for row in rows:
        label = str(row["gpu_layers"]) + (f" `{row['server_args']}`" if row["server_args"] else "")
        lines.append(f"| {label} | {row['blocks_on_cpu']} | {row['decode_tokens_per_s']:.1f} | "
                     f"{row['decode_ms_per_token']:.2f} | {row['prefill_ms_per_token']:.3f} | "
                     + " | ".join(f"{row[f'total_s_{lang}']:.3f}" for lang in languages)
                     + f" | {_number(row['server_vram_mib'], 0)} | {_number(row['cpu_cores'])} | "
                       f"{_number(row['gpu_util_pct'], 0)} | {_number(row['gpu_power_w'])} |")
    lines += ["", "GPU utilization and power are whole-GPU readings sampled every 100 ms; server VRAM is the "
              "server process alone. Requests run one at a time with a fixed output budget, so these numbers "
              "describe single-request latency, not throughput under load.", ""]
    return "\n".join(lines)


def analyze_layer_sweep(root: str | Path, output: str | Path) -> None:
    root, output = Path(root), Path(output)
    experiment, trials = load_trials(root)
    rows = summarize(experiment, trials)
    models = {metric: cost_model(trials, metric) for metric in ("decode_ms_per_token", "prefill_ms_per_token")}
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "trials.csv", trials)
    write_csv(output / "layers.csv", rows)
    (output / "cost-model.json").write_text(json.dumps(models, indent=2) + "\n", encoding="utf-8")
    (output / "report.md").write_text(report(experiment, rows, models), encoding="utf-8")
    print(f"Validated {len(trials)} trials. Report: {output / 'report.md'}")
