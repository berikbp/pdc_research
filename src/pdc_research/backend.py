"""llama.cpp commands, HTTP requests, and server process lifecycle."""

import json
import socket
import subprocess
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from .common import request


def binary(machine: dict, name: str) -> str:
    path = Path(machine["llama_bin"]) / name
    if not path.is_file():
        path = path.with_suffix(".exe")
    if not path.is_file():
        raise FileNotFoundError(f"Missing {name}; build llama.cpp first (see docs/setup.md)")
    return str(path.resolve())


def server_command(machine: dict, threads: int, layers: int) -> list[str]:
    return [
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
    ]


@contextmanager
def server(machine: dict, threads: int, layers: int, log: str | Path) -> Iterator[str]:
    # Do not accidentally benchmark an unrelated process already using this port.
    with socket.socket() as sock:
        if sock.connect_ex(("127.0.0.1", machine["port"])) == 0:
            raise RuntimeError(f"Port {machine['port']} already in use; stop that server or change port")
    command = server_command(machine, threads, layers)
    base = f"http://127.0.0.1:{machine['port']}"
    with Path(log).open("w", encoding="utf-8") as stream:
        stream.write(json.dumps(command) + "\n")
        stream.flush()
        process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT)
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


