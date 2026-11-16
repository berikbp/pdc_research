"""Find how many layers of a model fit in VRAM; run from the repo root.

Starts the server exactly as a measured run would, from the highest layer count
downwards, and sends the longest workload input with the full output budget.
A configuration counts as feasible only if that request completes. Nothing here
is a measurement: the output is the feasible range used to design an experiment.
"""

import argparse
import json
from pathlib import Path

from pdc_research.backend import completion, read_placement, server
from pdc_research.common import capture, load_rows, request, write_json
from pdc_research.config import load_configs
from pdc_research.telemetry import pid_file, server_gpu_memory


def attempt(machine: dict, experiment: dict, layers: int, text: str, log: Path) -> dict:
    threads = machine["threads"][0]
    try:
        with server(machine, threads, layers, log) as base:
            tokens = request(base, "/tokenize", {
                "content": text, "add_special": True, "parse_special": False,
            }, machine["request_timeout_seconds"])["tokens"]
            response, elapsed_ms = completion(base, tokens, experiment, machine["request_timeout_seconds"])
            timings = response["timings"]
            return {"gpu_layers": layers, "status": "ok", "placement": read_placement(log),
                    "server_vram": server_gpu_memory(int(pid_file(log).read_text())),
                    "gpu_memory_used_mib": capture(
                        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"]
                    ).get("stdout", "").strip(),
                    "input_tokens": timings["prompt_n"],
                    "decode_ms_per_token": timings["predicted_ms"] / timings["predicted_n"],
                    "prefill_ms_per_token": timings["prompt_ms"] / timings["prompt_n"],
                    "elapsed_ms": elapsed_ms}
    except (OSError, ValueError, RuntimeError, KeyError) as exc:
        return {"gpu_layers": layers, "status": "failed", "error": str(exc)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--machine", required=True)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--layers", required=True, type=int, nargs="+", help="Layer counts to try, in order")
    parser.add_argument("--stop-after-successes", type=int, default=0,
                        help="Stop once this many counts have loaded (0 tries all)")
    parser.add_argument("--output", required=True, type=Path, help="New directory for logs and probe.json")
    args = parser.parse_args()
    machine, experiment = load_configs(args.machine, args.experiment)
    rows = load_rows(args.manifest)
    longest = max(rows, key=lambda row: len(row["text"].encode("utf-8")))
    args.output.mkdir(parents=True, exist_ok=False)
    results, successes = [], 0
    for layers in args.layers:
        result = attempt(machine, experiment, layers, longest["text"], args.output / f"g{layers:02d}.log")
        results.append(result)
        print(json.dumps(result), flush=True)
        write_json(args.output / "probe.json", {
            "machine": machine, "experiment_id": experiment["experiment_id"],
            "input": {"group_id": longest["group_id"], "language": longest["language"]},
            "results": results})
        successes += result["status"] == "ok"
        if args.stop_after_successes and successes >= args.stop_after_successes:
            break


if __name__ == "__main__":
    main()
