"""Probe concurrent requests on one server; run from the repo root.

For each slot count the server is started with `slots * context_per_slot` tokens of
context, and every passage of one language is sent through a closed loop of `slots`
clients. The probe reports the VRAM the server holds after loading (before any
request) and the throughput per language. It is a feasibility probe with one pass
per cell, not a measurement with repetitions.
"""

import argparse
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from pdc_research.backend import completion, read_placement, server
from pdc_research.common import load_rows, request, write_json
from pdc_research.config import load_configs
from pdc_research.telemetry import pid_file, sampler, server_gpu_memory


def run_language(base: str, cases: list[list[int]], slots: int, experiment: dict, timeout: float) -> dict:
    def one(tokens: list[int]) -> dict:
        response, elapsed_ms = completion(base, tokens, experiment, timeout)
        return {**response["timings"], "elapsed_ms": elapsed_ms}

    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=slots) as pool:
        results = list(pool.map(one, cases))
    wall_s = time.perf_counter() - started
    generated = sum(r["predicted_n"] for r in results)
    prompt = sum(r["prompt_n"] for r in results)
    return {"requests": len(results), "wall_s": wall_s,
            "requests_per_s": len(results) / wall_s,
            "generated_tokens_per_s": generated / wall_s,
            "all_tokens_per_s": (generated + prompt) / wall_s,
            "mean_input_tokens": prompt / len(results),
            "mean_latency_s": sum(r["elapsed_ms"] for r in results) / len(results) / 1000,
            "max_latency_s": max(r["elapsed_ms"] for r in results) / 1000}


def probe(machine: dict, experiment: dict, texts: dict[str, list[str]], slots: int, layers: int,
          context_per_slot: int, log: Path) -> dict:
    runtime = {**machine, "context_size": slots * context_per_slot}
    cell = {"slots": slots, "gpu_layers": layers, "context_size": runtime["context_size"]}
    extra = ("-np", str(slots), "--no-kv-unified")  # later options override the runner's -np 1
    try:
        with server(runtime, machine["threads"][0], layers, log, extra_args=extra) as base, \
                sampler(log) as telemetry:
            cell["placement"] = read_placement(log)
            cell["server_vram_after_load"] = server_gpu_memory(int(pid_file(log).read_text()))
            timeout = machine["request_timeout_seconds"]
            cell["languages"] = {}
            for language, passages in texts.items():
                cases = [request(base, "/tokenize", {"content": text, "add_special": True,
                                                     "parse_special": False}, timeout)["tokens"]
                         for text in passages]
                run_language(base, cases[:slots], slots, experiment, timeout)  # unmeasured warm-up
                mark = telemetry.mark()
                cell["languages"][language] = {
                    **run_language(base, cases, slots, experiment, timeout),
                    "telemetry": telemetry.since(mark)}
            cell["server_vram_after_requests"] = server_gpu_memory(int(pid_file(log).read_text()))
            cell["status"] = "ok"
    except (OSError, ValueError, RuntimeError, KeyError) as exc:
        cell.update(status="failed", error=str(exc))
    return cell


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--machine", required=True)
    parser.add_argument("--experiment", required=True, help="Supplies seed and output_tokens")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--languages", required=True, nargs="+")
    parser.add_argument("--slots", required=True, type=int, nargs="+")
    parser.add_argument("--gpu-layers", required=True, type=int, nargs="+",
                        help="One value for all slot counts, or one value per slot count")
    parser.add_argument("--context-per-slot", required=True, type=int)
    parser.add_argument("--output", required=True, type=Path, help="New directory for logs and probe.json")
    args = parser.parse_args()
    if len(args.gpu_layers) not in (1, len(args.slots)):
        parser.error("--gpu-layers needs one value or one per slot count")
    layers = args.gpu_layers * len(args.slots) if len(args.gpu_layers) == 1 else args.gpu_layers
    machine, experiment = load_configs(args.machine, args.experiment)
    rows = load_rows(args.manifest)
    texts = {language: [row["text"] for row in rows if row["language"] == language]
             for language in args.languages}
    if not all(texts.values()):
        parser.error("A requested language has no passages in the manifest")
    args.output.mkdir(parents=True, exist_ok=False)
    cells = []
    for slots, count in zip(args.slots, layers):
        cell = probe(machine, experiment, texts, slots, count, args.context_per_slot,
                     args.output / f"np{slots:02d}-g{count:02d}.log")
        cells.append(cell)
        summary = {language: round(values["generated_tokens_per_s"], 1)
                   for language, values in cell.get("languages", {}).items()}
        print(f"slots={slots} layers={count} {cell['status']} "
              f"vram={cell.get('server_vram_after_load')} generated tok/s={summary} "
              f"{cell.get('error', '')}", flush=True)
        write_json(args.output / "probe.json", {
            "machine": machine, "experiment_id": experiment["experiment_id"],
            "context_per_slot": args.context_per_slot, "cells": cells})


if __name__ == "__main__":
    main()
