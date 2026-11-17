"""llama.cpp commands, HTTP requests, and server process lifecycle."""

import json
import re
import socket
import subprocess
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from .common import request
from .telemetry import pid_file


def binary(machine: dict, name: str) -> str:
    path = Path(machine["llama_bin"]) / name
    if not path.is_file():
        path = path.with_suffix(".exe")
    if not path.is_file():
        raise FileNotFoundError(f"Missing {name}; build llama.cpp first (see docs/setup.md)")
    return str(path.resolve())


def server_command(machine: dict, threads: int, layers: int) -> list[str]:
    command = [
        binary(machine, "llama-server"),
        "-m", machine["model"],
        "-t", str(threads),
        "-tb", str(threads),
        "-ngl", str(layers),
        "-c", str(machine["context_size"]),
        "-b", str(machine["batch_size"]),
        "-ub", str(machine["ubatch_size"]),
        "-np", "1",
        "--host", "127.0.0.1",
        "--port", str(machine["port"]),
        "--fit", "off",
        "--no-context-shift",
        "-fa", "off",
        "-ctk", "f16",
        "-ctv", "f16",
        "-lv", "4",  # trace level: the lowest that reports layer placement
    ]
    for override in machine.get("tensor_overrides", []):
        command.extend(["-ot", override])
    return [*command, *machine.get("server_args", [])]


def read_placement(log: str | Path) -> dict:
    """Actual layer placement reported by the server, not the requested -ngl value.

    Only the "offloaded N/M" total is used. M counts the transformer blocks plus the
    output layer, and llama.cpp offloads from the end: the output layer first, then the
    last blocks, so blocks left on the CPU are the first ones. The separate "repeating
    layers" log line is not used because it reports the same count for 28/29 and 29/29.
    """
    try:
        text = Path(log).read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        return {"error": f"cannot read server log: {exc}"}
    total = re.search(r"offloaded (\d+)/(\d+) layers to GPU", text)
    if not total:
        return {"error": "server log has no layer placement report"}
    offloaded, layers = int(total[1]), int(total[2])
    output_on_gpu = "offloading output layer to GPU" in text
    if output_on_gpu != (offloaded >= 1):
        return {"error": "server log placement lines disagree about the output layer"}
    blocks_on_gpu = max(offloaded - 1, 0)
    return {"offloaded_layers": offloaded, "total_layers": layers,
            "blocks_on_gpu": blocks_on_gpu, "blocks_on_cpu": layers - 1 - blocks_on_gpu,
            "output_layer_on_gpu": output_on_gpu}


@contextmanager
def server(machine: dict, threads: int, layers: int, log: str | Path,
           *, extra_args: tuple[str, ...] = ()) -> Iterator[str]:
    # Do not accidentally benchmark an unrelated process already using this port.
    with socket.socket() as sock:
        if sock.connect_ex(("127.0.0.1", machine["port"])) == 0:
            raise RuntimeError(f"Port {machine['port']} already in use; stop that server or change port")
    command = [*server_command(machine, threads, layers), *extra_args]
    base = f"http://127.0.0.1:{machine['port']}"
    with Path(log).open("w", encoding="utf-8") as stream:
        stream.write(json.dumps(command) + "\n")
        stream.flush()
        process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT)
        pid_file(log).write_text(str(process.pid))  # lets telemetry attribute CPU and VRAM
        try:
            deadline = time.monotonic() + machine["startup_timeout_seconds"]
            while True:
                if process.poll() is not None:
                    raise RuntimeError(f"llama-server exited ({process.returncode}); see {log}")
                try:
                    request(base, "/health", timeout=2)
                    break
                except (OSError, ValueError):
                    if time.monotonic() > deadline:
                        raise TimeoutError(f"Server startup timed out; see {log}")
                    time.sleep(0.25)
            yield base
        finally:
            pid_file(log).unlink(missing_ok=True)
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()


def completion(base: str, tokens: list[int], experiment: dict, timeout: float) -> tuple[dict, float]:
    payload = {
        "prompt": tokens,
        "n_predict": experiment["output_tokens"],
        "temperature": 0,
        "seed": experiment["seed"],
        "ignore_eos": True,
        "cache_prompt": False,
        "stream": False,
    }
    start = time.perf_counter()
    response = request(base, "/completion", payload, timeout)
    elapsed = (time.perf_counter() - start) * 1000
    timings = response["timings"]
    if timings["predicted_n"] != experiment["output_tokens"]:
        raise ValueError("Output token count differs from fixed generation budget")
    if timings["prompt_n"] != len(tokens) or timings.get("cache_n", 0):
        raise ValueError("Prompt length mismatch or cached prompt detected")
    if response.get("truncated"):
        raise ValueError("Server truncated the input")
    return response, elapsed


