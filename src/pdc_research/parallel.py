"""Concurrent requests on one server: throughput per cell, language and round.

A cell is one server start: GPU layers, slot count and context per slot. The server
gets `slots * context_per_slot` tokens of context with a separate KV buffer per slot.
Each language of the cell is driven by a closed loop of `slots` clients over a fixed,
seeded request order whose length is a multiple of the slot count, so every wave is
full. One unmeasured wave precedes each measured language block. Cells are reshuffled
in every round and each (round, cell) is one immutable result file, which makes an
interrupted run resumable.
"""

import csv
import random
import statistics as st
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .backend import binary, completion, read_placement, server
from .checkpoints import atomic_json
from .common import load_rows, read_json, request
from .config import validate_machine
from .session import build_fingerprint, initialize_run
from .telemetry import pid_file, sampler, server_gpu_memory

CELL_FIELDS = ("id", "gpu_layers", "slots", "context_per_slot", "languages")


def validate_parallel(experiment: dict) -> None:
    for name in ("experiment_id", "seed", "output_tokens", "rounds", "requests_per_language", "threads", "cells"):
        if name not in experiment:
            raise ValueError(f"Parallel experiment missing: {name}")
    ids = [cell.get("id") for cell in experiment["cells"]]
    if not ids or len(set(ids)) != len(ids):
        raise ValueError("Cell IDs must be unique and present")
    for cell in experiment["cells"]:
        missing = [name for name in CELL_FIELDS if name not in cell]
        if missing:
            raise ValueError(f"Cell {cell.get('id')} missing: {', '.join(missing)}")
        if cell["slots"] < 1 or experiment["requests_per_language"] % cell["slots"]:
            raise ValueError(f"Cell {cell['id']}: requests_per_language must be a multiple of slots")
        if not cell["languages"]:
            raise ValueError(f"Cell {cell['id']} has no languages")


def request_order(passages: int, requests: int, seed: int) -> list[int]:
    """Indices of the passages to send: shuffled passes over all passages, cut to length."""
    rng, order = random.Random(seed), []
    while len(order) < requests:
        block = list(range(passages))
        rng.shuffle(block)
        order.extend(block)
    return order[:requests]


def drive(base: str, cases: list[list[int]], slots: int, experiment: dict, timeout: float) -> list[dict]:
    origin = time.perf_counter()

    def one(tokens: list[int]) -> dict:
        started = time.perf_counter() - origin
        response, elapsed_ms = completion(base, tokens, experiment, timeout)
        timings = response["timings"]
        return {"started_s": started, "elapsed_ms": elapsed_ms, "prompt_n": timings["prompt_n"],
                "predicted_n": timings["predicted_n"], "prompt_ms": timings["prompt_ms"],
                "predicted_ms": timings["predicted_ms"]}

    with ThreadPoolExecutor(max_workers=slots) as pool:
        return list(pool.map(one, cases))


def summarize_block(requests: list[dict]) -> dict:
    """Throughput of one closed-loop block; wall time runs from the first start to the last end."""
    wall_s = max(r["started_s"] + r["elapsed_ms"] / 1000 for r in requests) - min(r["started_s"] for r in requests)
    generated = sum(r["predicted_n"] for r in requests)
    prompt = sum(r["prompt_n"] for r in requests)
    return {"requests": len(requests), "wall_s": wall_s, "requests_per_s": len(requests) / wall_s,
            "generated_tokens_per_s": generated / wall_s, "all_tokens_per_s": (generated + prompt) / wall_s,
            "mean_input_tokens": prompt / len(requests),
            "mean_latency_s": st.mean(r["elapsed_ms"] for r in requests) / 1000,
            "max_latency_s": max(r["elapsed_ms"] for r in requests) / 1000}


def run_cell(machine: dict, experiment: dict, cell: dict, texts: dict[str, list[str]],
             round_: int, log: Path) -> dict:
    runtime = {**machine, "context_size": cell["slots"] * cell["context_per_slot"]}
    extra = ("-np", str(cell["slots"]), "--no-kv-unified")  # later options override the runner's -np 1
    timeout = machine["request_timeout_seconds"]
    record = {"cell": cell, "round": round_, "context_size": runtime["context_size"], "languages": {}}
    languages = list(cell["languages"])
    random.Random(f"{experiment['seed']}-{round_}-{cell['id']}").shuffle(languages)
    try:
        with server(runtime, experiment["threads"], cell["gpu_layers"], log, extra_args=extra) as base, \
                sampler(log) as telemetry:
            pid = int(pid_file(log).read_text())
            record["placement"] = read_placement(log)
            record["server_vram_after_load"] = server_gpu_memory(pid)
            for position, language in enumerate(languages):
                tokens = [request(base, "/tokenize", {"content": text, "add_special": True,
                                                      "parse_special": False}, timeout)["tokens"]
                          for text in texts[language]]
                if max(map(len, tokens)) + experiment["output_tokens"] > cell["context_per_slot"]:
                    raise ValueError(f"{language} does not fit in {cell['context_per_slot']} tokens per slot")
                order = request_order(len(tokens), experiment["requests_per_language"],
                                      experiment["seed"] + 1000 * round_)
                cases = [tokens[index] for index in order]
                drive(base, cases[:cell["slots"]], cell["slots"], experiment, timeout)  # unmeasured wave
                mark = telemetry.mark()
                requests = drive(base, cases, cell["slots"], experiment, timeout)
                record["languages"][language] = {
                    "position": position, "passage_order": order, **summarize_block(requests),
                    "telemetry": telemetry.since(mark), "requests_detail": requests}
            record["server_vram_after_requests"] = server_gpu_memory(pid)
            record["status"] = "ok"
    except (OSError, ValueError, RuntimeError, KeyError) as exc:
        record.update(status="failed", error=str(exc))
    return record


def run_parallel(machine_path: str, experiment_path: str, manifest: str, output: str, *,
                 resume: bool = False) -> None:
    machine, experiment = read_json(machine_path), read_json(experiment_path)
    validate_parallel(experiment)
    validate_machine({**machine, "threads": [experiment["threads"]]})
    root = Path(output)
    if root.exists() and not resume:
        raise FileExistsError("Output exists; pass --resume")
    rows = load_rows(manifest)
    wanted = sorted({language for cell in experiment["cells"] for language in cell["languages"]})
    texts = {language: [row["text"] for row in rows if row["language"] == language] for language in wanted}
    if not all(texts.values()):
        raise ValueError("A requested language has no passages in the manifest")
    executable = binary(machine, "llama-server")
    identity = build_fingerprint(machine, experiment, manifest, executable)
    identity["execution_order"] = "parallel-rounds-v1"
    root.mkdir(parents=True, exist_ok=True)
    initialize_run(root, identity, rows, executable, resume)
    total = experiment["rounds"] * len(experiment["cells"])
    done = 0
    for round_ in range(1, experiment["rounds"] + 1):
        cells = list(experiment["cells"])
        random.Random(experiment["seed"] + round_).shuffle(cells)
        for cell in cells:
            done += 1
            path = root / "cells" / f"round-{round_}-{cell['id']}.json"
            if path.exists() and read_json(path).get("status") == "ok":
                continue
            log = root / "logs" / f"round-{round_}-{cell['id']}-{int(time.time())}.log"
            log.parent.mkdir(parents=True, exist_ok=True)
            record = run_cell(machine, experiment, cell, texts, round_, log)
            atomic_json(path, record)
            rates = {language: round(block["generated_tokens_per_s"], 1)
                     for language, block in record["languages"].items()}
            print(f"[{done}/{total}] round {round_} {cell['id']}: {record['status']} {rates} "
                  f"{record.get('error', '')}", flush=True)
    atomic_json(root / "status.json", {"phase": "complete", "finished": True})


def load_cells(root: Path) -> tuple[dict, list[dict]]:
    experiment = read_json(root / "metadata.json")["fingerprint"]["experiment"]
    rows = []
    for cell in experiment["cells"]:
        for round_ in range(1, experiment["rounds"] + 1):
            path = root / "cells" / f"round-{round_}-{cell['id']}.json"
            if not path.exists():
                raise ValueError(f"Incomplete run: missing {path.name}")
            record = read_json(path)
            if record["status"] != "ok":
                raise ValueError(f"Failed cell: {path.name}: {record.get('error')}")
            placement = record["placement"]
            if "error" in placement or placement["offloaded_layers"] != min(cell["gpu_layers"],
                                                                           placement["total_layers"]):
                raise ValueError(f"Placement mismatch: {path.name}")
            for language in cell["languages"]:
                block = record["languages"][language]
                if block["requests"] != experiment["requests_per_language"]:
                    raise ValueError(f"Wrong request count: {path.name} {language}")
                telemetry = block["telemetry"]
                rows.append({
                    "cell": cell["id"], "round": round_, "language": language,
                    "gpu_layers": cell["gpu_layers"], "blocks_on_cpu": placement["blocks_on_cpu"],
                    "slots": cell["slots"], "context_per_slot": cell["context_per_slot"],
                    "context_size": record["context_size"],
                    "server_vram_mib": record["server_vram_after_load"].get("vram_mib"),
                    **{key: block[key] for key in ("requests", "wall_s", "requests_per_s",
                                                   "generated_tokens_per_s", "all_tokens_per_s",
                                                   "mean_input_tokens", "mean_latency_s", "max_latency_s")},
                    "gpu_util_pct": telemetry["gpu"].get("gpu_util_mean_pct"),
                    "gpu_power_w": telemetry["gpu"].get("power_mean_w"),
                    "cpu_cores": telemetry["process"].get("cpu_cores_mean")})
    return experiment, rows


def aggregate(experiment: dict, rows: list[dict]) -> list[dict]:
    """One row per cell and language: means over rounds and the spread between them."""
    table = []
    for cell in experiment["cells"]:
        for language in cell["languages"]:
            own = [r for r in rows if r["cell"] == cell["id"] and r["language"] == language]
            rates = [r["generated_tokens_per_s"] for r in own]
            first = own[0]
            entry = {key: first[key] for key in ("cell", "language", "gpu_layers", "blocks_on_cpu", "slots",
                                                 "context_per_slot", "context_size", "server_vram_mib",
                                                 "mean_input_tokens")}
            entry["rounds"] = len(own)
            for key in ("generated_tokens_per_s", "requests_per_s", "all_tokens_per_s", "mean_latency_s"):
                entry[key] = st.mean(r[key] for r in own)
            entry["round_spread_pct"] = 100 * (max(rates) - min(rates)) / st.mean(rates)
            for key in ("gpu_util_pct", "gpu_power_w", "cpu_cores"):
                present = [r[key] for r in own if r[key] is not None]
                entry[key] = st.mean(present) if present else None
            table.append(entry)
    return table


def report(experiment: dict, table: list[dict]) -> str:
    lines = [f"# Concurrent requests: {experiment['experiment_id']}", "",
             f"{experiment['requests_per_language']} requests per cell, language and round in a closed loop of "
             f"`slots` clients; {experiment['rounds']} rounds, each with a fresh server start; "
             f"{experiment['output_tokens']} generated tokens per request. Throughput is generated tokens "
             "per second of wall time, so prompt processing is inside it. Spread is the range between "
             "rounds relative to their mean.", "",
             "| Cell | GPU layers | Blocks on CPU | Slots | Context/slot | Server VRAM, MiB | Language | "
             "Input tokens | Generated tok/s | Requests/s | Latency, s | Spread, % | CPU cores | GPU util, % |",
             "|---|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in table:
        optional = [("n/a" if row[key] is None else f"{row[key]:.1f}") for key in ("cpu_cores", "gpu_util_pct")]
        lines.append(
            f"| {row['cell']} | {row['gpu_layers']} | {row['blocks_on_cpu']} | {row['slots']} | "
            f"{row['context_per_slot']} | {row['server_vram_mib']} | {row['language']} | "
            f"{row['mean_input_tokens']:.1f} | {row['generated_tokens_per_s']:.1f} | "
            f"{row['requests_per_s']:.3f} | {row['mean_latency_s']:.2f} | {row['round_spread_pct']:.1f} | "
            f"{optional[0]} | {optional[1]} |")
    return "\n".join(lines) + "\n"


def analyze_parallel(root: str | Path, output: str | Path) -> None:
    root, output = Path(root), Path(output)
    experiment, rows = load_cells(root)
    table = aggregate(experiment, rows)
    output.mkdir(parents=True, exist_ok=True)
    for name, data in (("rounds.csv", rows), ("cells.csv", table)):
        with (output / name).open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(data[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(data)
    (output / "report.md").write_text(report(experiment, table), encoding="utf-8")
    print(f"Validated {len(rows)} blocks. Report: {output / 'report.md'}")
