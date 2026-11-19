"""Stage 2: NEO (CPU offload) against the same engine GPU-only and vLLM, under load.

A cell is one server start: system, GPU KV-cache budget in blocks and load (the number
of concurrent closed-loop clients). Every language of the experiment runs on the same
server start, in a seeded order that changes between rounds. Each language block is
preceded by unmeasured warm-up waves; the measured block has a whole number of waves
of `load` requests. One immutable file per (round, cell) makes a run resumable.

Requests are pre-tokenized (scripts/tokenize_workload.py), streamed, and must return
exactly `output_tokens` tokens. NEO ignores EOS by design; vLLM is told to.
"""

import csv
import http.client
import json
import math
import os
import random
import re
import signal
import socket
import statistics as st
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from datetime import datetime, timezone
from hashlib import sha256 as sha256_bytes
from pathlib import Path
from collections.abc import Iterator

from .checkpoints import atomic_json
from .common import append_row, capture, read_json, sha256
from .parallel import request_order
from .telemetry import Sampler, process_tree, run_conditions, server_gpu_memory

# neo: NEO as shipped (CPU launch time 0.8 ms, the authors' constant); neo_tuned: NEO with the launch
# time measured on this machine (calibrate_neo); neo_base: the same engine GPU-only; vllm: reference.
SYSTEMS = ("neo", "neo_tuned", "neo_base", "vllm")
HYBRID = ("neo", "neo_tuned")
NEO_SYSTEMS = HYBRID + ("neo_base",)
AUTHORS_CPU_LAUNCH_MS = 0.8
# Tables NEO builds at start-up (ModelProfiler.init_profile_tables); lnch.json only with --profile-cpu-launch.
NEO_PROFILE_FILES = ("linr.json", "pref.json", "gdec.json", "cdec.json")
LAUNCH_PROFILE = "lnch.json"
EXPERIMENT_FIELDS = ("experiment_id", "seed", "output_tokens", "rounds", "systems", "memory_blocks", "loads",
                     "languages", "min_requests", "min_waves", "warmup_waves", "block_size", "max_model_len",
                     "max_tokens_in_batch", "max_num_seqs")
MACHINE_FIELDS = ("machine_id", "python", "vllm", "neo_dir", "model", "neo_library", "neo_profile_dir",
                  "gpu_index", "port", "swap_gb", "gpu_memory_utilization", "num_layers",
                  "startup_timeout_seconds", "request_timeout_seconds", "min_free_ram_gb")


# ----------------------------------------------------------------------------- plan

def memory_levels(experiment: dict, system: str) -> list[int | None]:
    """KV budgets measured for one system: its own list if the experiment narrows it, else the shared one."""
    return experiment.get("memory_blocks_by_system", {}).get(system, experiment["memory_blocks"])


def validate_serving(experiment: dict) -> None:
    missing = [name for name in EXPERIMENT_FIELDS if name not in experiment]
    if missing:
        raise ValueError(f"Serving experiment missing: {', '.join(missing)}")
    unknown = set(experiment["systems"]) - set(SYSTEMS)
    if unknown:
        raise ValueError(f"Unknown systems: {sorted(unknown)}")
    stray = set(experiment.get("memory_blocks_by_system", {})) - set(experiment["systems"])
    if stray:
        raise ValueError(f"memory_blocks_by_system names systems not in the experiment: {sorted(stray)}")
    if not experiment["loads"] or min(experiment["loads"]) < 1:
        raise ValueError("Loads must be positive")
    if max(experiment["loads"]) > experiment["max_num_seqs"]:
        raise ValueError("max_num_seqs must admit the largest load")
    for blocks in {b for system in experiment["systems"] for b in memory_levels(experiment, system)}:
        if blocks is None:
            continue
        neo_blocks = blocks * 32 // 33  # NEO's share; checked exactly against num_layers at run time
        if neo_blocks * experiment["block_size"] < experiment["max_tokens_in_batch"]:
            raise ValueError(f"{blocks} blocks cannot hold max_tokens_in_batch; NEO refuses to start")
        if blocks * experiment["block_size"] < experiment["max_model_len"]:
            raise ValueError(f"{blocks} blocks cannot hold one request of max_model_len")


def requests_for(load: int, experiment: dict) -> int:
    """Measured requests per language: whole waves, at least `min_waves` and `min_requests`."""
    wanted = max(experiment["min_requests"], experiment["min_waves"] * load)
    return math.ceil(wanted / load) * load


def memory_label(blocks: int | None) -> str:
    return "free" if blocks is None else f"b{blocks:04d}"


def cells(experiment: dict) -> list[dict]:
    return [{"id": f"{system}-{memory_label(blocks)}-c{load:02d}", "system": system, "blocks": blocks,
             "load": load}
            for system in experiment["systems"] for blocks in memory_levels(experiment, system)
            for load in experiment["loads"]]


def round_order(experiment: dict, round_: int) -> list[dict]:
    order = cells(experiment)
    random.Random(f"{experiment['seed']}-cells-{round_}").shuffle(order)
    return order


def language_order(experiment: dict, round_: int, cell_id: str) -> list[str]:
    languages = list(experiment["languages"])
    random.Random(f"{experiment['seed']}-{round_}-{cell_id}").shuffle(languages)
    return languages


def check_cases(cases: dict, experiment: dict) -> None:
    for language in experiment["languages"]:
        if not cases["cases"].get(language):
            raise ValueError(f"No tokenized passages for {language}")
        longest = max(len(case["tokens"]) for case in cases["cases"][language])
        if longest + experiment["output_tokens"] > experiment["max_model_len"]:
            raise ValueError(f"{language}: {longest} + {experiment['output_tokens']} tokens exceed max_model_len")
        if longest > experiment["max_tokens_in_batch"]:
            raise ValueError(f"{language}: prompt longer than max_tokens_in_batch")


# --------------------------------------------------------------------------- server

def gpu_blocks_requested(system: str, blocks: int | None, machine: dict) -> int | None:
    """KV blocks the system is asked for; NEO gives one layer's share to CPU prefill, as in its scripts."""
    if blocks is None:
        return None
    if system in HYBRID:
        layers = machine["num_layers"]
        return blocks * layers // (layers + 1)
    return blocks


def blocks_per_sequence(experiment: dict) -> int:
    """NEO profiles one prefill of `max_tokens_in_batch` tokens, so a sequence must hold that many too."""
    return math.ceil(max(experiment["max_model_len"], experiment["max_tokens_in_batch"]) / experiment["block_size"])


def server_command(system: str, blocks: int | None, machine: dict, experiment: dict,
                   extra: tuple[str, ...] = ()) -> list[str]:
    requested = gpu_blocks_requested(system, blocks, machine)
    if system == "vllm":
        command = [machine["vllm"], "serve", machine["model"], "--host", "127.0.0.1",
                   "--port", str(machine["port"]), "--dtype", "float16", "--seed", "0",
                   "--block-size", str(experiment["block_size"]),
                   "--max-model-len", str(experiment["max_model_len"]),
                   "--max-num-seqs", str(experiment["max_num_seqs"]),
                   "--max-num-batched-tokens", str(experiment["max_tokens_in_batch"]),
                   "--gpu-memory-utilization", str(machine["gpu_memory_utilization"]),
                   "--no-enable-prefix-caching", "--enforce-eager"]
        if requested is not None:
            command += ["--num-gpu-blocks-override", str(requested)]
        return command + machine.get("vllm_args", [])
    block_size = experiment["block_size"]
    command = [machine["python"], "-m", "swiftllm.server.api_server", "--host", "127.0.0.1",
               "--port", str(machine["port"]), "--model-path", machine["model"],
               "--block-size", str(block_size),
               "--max-blocks-per-seq", str(blocks_per_sequence(experiment)),
               "--max-seqs-in-block-table", str(experiment["max_num_seqs"]),
               "--max-batch-size", str(experiment["max_num_seqs"]),
               "--max-tokens-in-batch", str(experiment["max_tokens_in_batch"]),
               "--tensor-parallel-degree", "1", "--swap-space", str(machine["swap_gb"]),
               "--gpu-mem-utilization", str(machine["gpu_memory_utilization"]),
               "--library-path", machine["neo_library"],
               "--profile-result-path", machine["neo_profile_dir"].rstrip("/") + "/"]
    if requested is not None:
        command += ["--num-gpu-blocks-override", str(requested)]
    if system == "neo_tuned":
        if machine.get("neo_cpu_launch_ms") is None:
            raise ValueError("neo_tuned needs neo_cpu_launch_ms in the machine profile; run calibrate-neo")
        command += ["--cpu-launch-time", str(machine["neo_cpu_launch_ms"])]
    return command + (["--extra-layer-for-cprf"] if system in HYBRID else ["--always-use-gpu"]) + list(extra)


def server_environment(system: str, machine: dict) -> dict:
    env = {**os.environ, "CUDA_DEVICE_ORDER": "PCI_BUS_ID", "CUDA_VISIBLE_DEVICES": str(machine["gpu_index"]),
           "TOKENIZERS_PARALLELISM": "false"}
    if system != "vllm":
        env["PYTHONPATH"] = f"{machine['neo_dir']}:{machine['neo_dir']}/csrc"
        if machine.get("omp_threads"):
            env["OMP_NUM_THREADS"] = str(machine["omp_threads"])
        if machine.get("omp_cpu_affinity"):
            # Careful: libgomp also binds the INITIAL thread to the first CPU of the list at start-up, and every
            # thread and Ray process started later inherits it, so the whole server ends up on one core.
            # The lab machines therefore leave this unset (as the NEO authors do: numactl -N 0, no pinning).
            env["GOMP_CPU_AFFINITY"] = machine["omp_cpu_affinity"]
    return env


REPORTED = {
    "neo": (re.compile(r"Number of GPU blocks: (\d+)"), 1),
    "neo_tuned": (re.compile(r"Number of GPU blocks: (\d+)"), 1),
    "neo_base": (re.compile(r"Number of GPU blocks: (\d+)"), 1),
    "vllm": (re.compile(r"GPU KV cache size: ([\d,]+) tokens"), None),
}


def reported_kv(system: str, log_text: str, block_size: int) -> dict:
    """KV capacity the system itself reports after start, in blocks."""
    pattern, _ = REPORTED[system]
    found = pattern.findall(log_text)
    if not found:
        return {"error": "KV capacity not found in the server log"}
    value = int(found[-1].replace(",", ""))
    return {"gpu_blocks": value if system != "vllm" else value // block_size, "raw": found[-1]}


def free_ram_gb() -> float | None:
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / 1024 / 1024
    except OSError:
        return None
    return None


def get_json(port: int, path: str, timeout: float = 5) -> dict | str:
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
    try:
        connection.request("GET", path)
        response = connection.getresponse()
        body = response.read().decode("utf-8", "replace")
        if response.status != 200:
            raise OSError(f"GET {path}: HTTP {response.status}")
        try:
            return json.loads(body)
        except ValueError:
            return body
    finally:
        connection.close()


@contextmanager
def serving(system: str, blocks: int | None, machine: dict, experiment: dict, log: Path,
            extra: tuple[str, ...] = ()) -> Iterator[int]:
    """Start one server in its own process group, wait until it answers, always stop it."""
    port = machine["port"]
    with socket.socket() as sock:
        if sock.connect_ex(("127.0.0.1", port)) == 0:
            raise RuntimeError(f"Port {port} already in use; stop that server first")
    if system != "vllm":
        free = free_ram_gb()
        if free is not None and free < machine["swap_gb"] + machine["min_free_ram_gb"]:
            raise RuntimeError(f"Only {free:.0f} GB RAM available for {machine['swap_gb']} GB of swap space")
    command = server_command(system, blocks, machine, experiment, extra)
    ready = "/stats" if system != "vllm" else "/health"
    with log.open("w", encoding="utf-8") as stream:
        stream.write(json.dumps(command) + "\n")
        stream.flush()
        process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT, cwd=machine["neo_dir"],
                                   env=server_environment(system, machine), start_new_session=True)
        try:
            deadline = time.monotonic() + machine["startup_timeout_seconds"]
            while True:
                if process.poll() is not None:
                    raise RuntimeError(f"{system} exited ({process.returncode}); see {log}")
                try:
                    get_json(port, ready, timeout=2)
                    break
                except (OSError, http.client.HTTPException):
                    if time.monotonic() > deadline:
                        raise TimeoutError(f"{system} startup timed out; see {log}")
                    time.sleep(1)
            yield process.pid
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)  # vLLM keeps its engine in a child process
                try:
                    process.wait(timeout=60)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
            time.sleep(3)  # let the driver release VRAM before the next start


# --------------------------------------------------------------------------- client

def completion_payload(system: str, tokens: list[int], experiment: dict, machine: dict) -> dict:
    payload = {"prompt": tokens, "max_tokens": experiment["output_tokens"], "stream": True}
    if system == "vllm":
        payload.update(model=machine["model"], temperature=0.0, ignore_eos=True,
                       stream_options={"include_usage": True})
    return payload


def parse_stream(system: str, lines: Iterator[bytes], clock) -> dict:
    """Output token count, time of the first token and an output digest from a streamed response."""
    first, count, usage, digest = None, 0, None, sha256_bytes()
    for raw in lines:
        line = raw.decode("utf-8").strip()
        if not line:
            continue
        if system == "vllm":
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            chunk = json.loads(line[5:])
            if chunk.get("usage"):
                usage = chunk["usage"]["completion_tokens"]
            choices = chunk.get("choices") or []
            if not choices:
                continue
            digest.update(choices[0].get("text", "").encode("utf-8"))
            count += 1
        else:  # NEO streams one token ID per line
            digest.update((line + ",").encode())
            count += 1
        if first is None:
            first = clock()
    return {"first": first, "output_tokens": usage if usage is not None else count,
            "output_sha256": digest.hexdigest()}


def stream_completion(system: str, port: int, payload: dict, timeout: float) -> dict:
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
    started = time.perf_counter()
    try:
        connection.request("POST", "/v1/completions", body=json.dumps(payload),
                           headers={"Content-Type": "application/json"})
        response = connection.getresponse()
        if response.status != 200:
            raise RuntimeError(f"HTTP {response.status}: {response.read()[:300]!r}")
        parsed = parse_stream(system, iter(response.readline, b""), time.perf_counter)
    finally:
        connection.close()
    ended = time.perf_counter()
    return {"started": started, "first": parsed["first"] or ended, "ended": ended,
            "output_tokens": parsed["output_tokens"], "output_sha256": parsed["output_sha256"]}


def drive(send, cases: list[dict], load: int) -> list[dict]:
    """Closed loop: `load` clients, each sends its next request when the previous one ends."""
    origin = time.perf_counter()

    def one(case: dict) -> dict:
        result = send(case["tokens"])
        return {"group_id": case["group_id"], "input_tokens": len(case["tokens"]),
                "started_s": result["started"] - origin, "ttft_s": result["first"] - result["started"],
                "elapsed_s": result["ended"] - result["started"], "output_tokens": result["output_tokens"],
                "output_sha256": result["output_sha256"]}

    with ThreadPoolExecutor(max_workers=load) as pool:
        return list(pool.map(one, cases))


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, max(0, math.ceil(fraction * len(ordered)) - 1))]


def summarize_requests(requests: list[dict], output_tokens: int) -> dict:
    """Throughput over the whole block and over its steady part (first and last 10% of completions cut)."""
    bad = [r for r in requests if r["output_tokens"] != output_tokens]
    if bad:
        raise ValueError(f"{len(bad)} responses did not return exactly {output_tokens} tokens")
    ends = sorted(r["started_s"] + r["elapsed_s"] for r in requests)
    wall = ends[-1] - min(r["started_s"] for r in requests)
    cut = len(ends) // 10
    kept = ends[cut:len(ends) - cut]
    steady = (len(kept) - 1) / (kept[-1] - kept[0]) if len(kept) > 1 and kept[-1] > kept[0] else None
    latency = [r["elapsed_s"] for r in requests]
    ttft = [r["ttft_s"] for r in requests]
    decode = [(r["elapsed_s"] - r["ttft_s"]) / (output_tokens - 1) for r in requests] if output_tokens > 1 else [0.0]
    return {"requests": len(requests), "wall_s": wall, "requests_per_s": len(requests) / wall,
            "generated_tokens_per_s": len(requests) * output_tokens / wall,
            "steady_generated_tokens_per_s": None if steady is None else steady * output_tokens,
            "mean_input_tokens": st.mean(r["input_tokens"] for r in requests),
            "mean_latency_s": st.mean(latency), "p50_latency_s": percentile(latency, 0.5),
            "p95_latency_s": percentile(latency, 0.95), "max_latency_s": max(latency),
            "mean_ttft_s": st.mean(ttft), "p95_ttft_s": percentile(ttft, 0.95),
            "mean_time_per_output_token_ms": 1000 * st.mean(decode)}


def counters(system: str, port: int) -> dict:
    """NEO: cumulative work counters (GET /stats). vLLM: preemptions from Prometheus metrics."""
    try:
        if system != "vllm":
            return get_json(port, "/stats")
        text = get_json(port, "/metrics")
        total = sum(float(line.rsplit(" ", 1)[1]) for line in str(text).splitlines()
                    if line.startswith("vllm:num_preemptions"))
        return {"preemptions": total}
    except (OSError, ValueError, IndexError, http.client.HTTPException) as exc:
        return {"error": f"counters unavailable: {exc}"}


def counter_delta(before: dict, after: dict) -> dict:
    if "error" in before or "error" in after:
        return {"error": before.get("error") or after.get("error")}
    settings = ("cpu_queue_max", "cpu_launch_time_ms")
    delta = {key: after[key] - before[key] for key in after if key not in settings and key in before}
    if "cpu_queue_max" in after:
        delta["cpu_queue_max_since_start"] = after["cpu_queue_max"]
    if "cpu_launch_time_ms" in after:
        delta["cpu_launch_time_ms"] = after["cpu_launch_time_ms"]
    decode = delta.get("gpu_decode_tokens", 0) + delta.get("cpu_decode_tokens", 0)
    if "cpu_decode_tokens" in delta:
        delta["cpu_decode_share"] = delta["cpu_decode_tokens"] / decode if decode else 0.0
        iterations = delta.get("iterations", 0)
        delta["pipelined_share"] = delta["pipelined_iterations"] / iterations if iterations else 0.0
    return delta


# ----------------------------------------------------------------------------- run

def language_cases(cases: dict, language: str, count: int, seed: int) -> tuple[list[int], list[dict]]:
    passages = cases["cases"][language]
    order = request_order(len(passages), count, seed)
    return order, [passages[index] for index in order]


def cpu_affinity(pid: int) -> dict:
    """CPUs the server process and its threads may run on (catches accidental pinning)."""
    try:
        threads = {}
        for task in Path(f"/proc/{pid}/task").iterdir():
            cpus = next(line.split(":", 1)[1].strip() for line in (task / "status").read_text().splitlines()
                        if line.startswith("Cpus_allowed_list"))
            threads[cpus] = threads.get(cpus, 0) + 1
        return {"process": sorted(os.sched_getaffinity(pid)), "threads_by_cpu_list": threads}
    except OSError as exc:
        return {"error": str(exc)}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def run_cell(machine: dict, experiment: dict, cell: dict, cases: dict, round_: int, log: Path) -> dict:
    system, load = cell["system"], cell["load"]
    record = {"cell": cell, "round": round_, "started_utc": datetime.now(timezone.utc).isoformat(),
              "gpu_blocks_requested": gpu_blocks_requested(system, cell["blocks"], machine), "languages": {}}
    measured = requests_for(load, experiment)
    port = machine["port"]
    timeout = machine["request_timeout_seconds"]

    def send(tokens: list[int]) -> dict:
        return stream_completion(system, port, completion_payload(system, tokens, experiment, machine), timeout)

    try:
        with serving(system, cell["blocks"], machine, experiment, log) as pid:
            record["kv_reported"] = reported_kv(system, log.read_text(errors="replace"), experiment["block_size"])
            record["server_vram_after_load"] = server_gpu_memory(pid, process_tree(pid))
            record["server_cpu_affinity"] = cpu_affinity(pid)
            record["conditions"] = run_conditions()
            languages = language_order(experiment, round_, cell["id"])
            seed = experiment["seed"] + 1000 * round_
            # One unmeasured wave of mixed languages after the start, then per-language warm-up waves.
            mixed = [language_cases(cases, language, load, seed)[1][index // len(languages)]
                     for index, language in zip(range(load), languages * load)]
            drive(send, mixed, load)
            with Sampler(pid, gpu_index=machine["gpu_index"], tree=True) as telemetry:
                for position, language in enumerate(languages):
                    order, block = language_cases(cases, language, measured, seed)
                    warm = experiment["warmup_waves"] * load
                    warmup_started = utc_now()
                    drive(send, [block[index % len(block)] for index in range(warm)], load)
                    mark, before = telemetry.mark(), counters(system, port)
                    measured_started = utc_now()
                    requests = drive(send, block, load)
                    measured_finished = utc_now()
                    work = counter_delta(before, counters(system, port))
                    # UTC bounds let the per-second journals (lab_monitor, lab_memmon) be cut per language.
                    record["languages"][language] = {
                        "position": position, "passage_order": order, "warmup_started_utc": warmup_started,
                        "measured_started_utc": measured_started, "measured_finished_utc": measured_finished,
                        **summarize_requests(requests, experiment["output_tokens"]),
                        "work": work, "telemetry": telemetry.since(mark), "requests_detail": requests}
            record["server_vram_after_requests"] = server_gpu_memory(pid, process_tree(pid))
            record["status"] = "ok"
    except (OSError, ValueError, RuntimeError, KeyError, http.client.HTTPException) as exc:
        record.update(status="failed", error=f"{type(exc).__name__}: {exc}")
    record["finished_utc"] = datetime.now(timezone.utc).isoformat()
    return record


def fingerprint(machine: dict, experiment: dict, cases_path: str) -> dict:
    model = Path(machine["model"])
    profile = Path(machine["neo_profile_dir"])
    missing = [name for name in NEO_PROFILE_FILES if not (profile / name).is_file()]
    if missing and set(NEO_SYSTEMS) & set(experiment["systems"]):
        raise ValueError(f"NEO profile incomplete in {profile}: {missing}. Run calibrate-neo first; "
                         "a profile built during a measured cell would change NEO's decisions mid-run.")
    files = list(NEO_PROFILE_FILES)
    if "neo_tuned" in experiment["systems"]:
        measured = measured_cpu_launch_ms(profile)
        if machine.get("neo_cpu_launch_ms") is None or abs(machine["neo_cpu_launch_ms"] - measured) > 1e-6:
            raise ValueError(f"neo_cpu_launch_ms must be the value measured by calibrate-neo ({measured:.6f}), "
                             f"not {machine.get('neo_cpu_launch_ms')}")
        files.append(LAUNCH_PROFILE)
    cases = read_json(cases_path)
    tokenizer = model / "tokenizer.json"
    if sha256(tokenizer) != cases["tokenizer"]["tokenizer_json_sha256"]:
        raise ValueError("Tokenized workload was made with a different tokenizer than the served model")
    return {
        "schema_version": 2, "execution_order": "serving-rounds-v1",
        "machine": machine, "experiment": experiment, "cases_sha256": sha256(cases_path),
        "model_files_sha256": {path.name: sha256(path) for path in sorted(model.glob("*.safetensors"))},
        "model_config_sha256": sha256(model / "config.json"),
        "neo_library_sha256": sha256(machine["neo_library"]),
        "neo_profile_sha256": profile_hashes(profile, files) if not missing else {},
        "neo_commit": capture(["git", "-C", machine["neo_dir"], "rev-parse", "HEAD"]).get("stdout", "").strip(),
        "neo_dirty": capture(["git", "-C", machine["neo_dir"], "status", "--porcelain", "--untracked-files=no"]
                             ).get("stdout", "").strip(),
        "vllm_version": capture([machine["python"], "-c", "import vllm; print(vllm.__version__)"]
                                ).get("stdout", "").strip(),
    }


def measured_cpu_launch_ms(profile: Path) -> float:
    """Mean of NEO's launch-time table, exactly as its profiler returns it to the scheduler."""
    path = profile / LAUNCH_PROFILE
    if not path.is_file():
        raise ValueError(f"No {LAUNCH_PROFILE} in {profile}; run calibrate-neo first")
    times = read_json(path)["T_list"]
    flat = [value for item in times for value in (item if isinstance(item, list) else [item])]
    return st.mean(flat)


def profile_hashes(profile: Path, names) -> dict:
    return {name: sha256(profile / name) for name in names}


def record_conditions(root: Path, event: str) -> None:
    with (root / "conditions.jsonl").open("a", encoding="utf-8") as stream:
        append_row(stream, {"event": event, **run_conditions()})


def run_serving(machine_path: str, experiment_path: str, cases_path: str, output: str, *,
                resume: bool = False, cell_runner=run_cell, identity: dict | None = None,
                until_round: int | None = None) -> dict:
    machine, experiment = read_json(machine_path), read_json(experiment_path)
    validate_serving(experiment)
    missing = [name for name in MACHINE_FIELDS if name not in machine]
    if missing:
        raise ValueError(f"Machine profile missing: {', '.join(missing)}")
    cases = read_json(cases_path)
    check_cases(cases, experiment)
    root = Path(output)
    if root.exists() and not resume:
        raise FileExistsError("Output exists; pass --resume")
    identity = identity if identity is not None else fingerprint(machine, experiment, cases_path)
    root.mkdir(parents=True, exist_ok=True)
    if resume:
        if read_json(root / "metadata.json")["fingerprint"] != identity:
            raise ValueError("Resume refused: machine, experiment, model, NEO, profile or workload changed")
        (root / "STOP").unlink(missing_ok=True)
    else:
        atomic_json(root / "metadata.json", {"fingerprint": identity,
                                             "started_utc": datetime.now(timezone.utc).isoformat(),
                                             "git": capture(["git", "rev-parse", "HEAD"]).get("stdout", "").strip(),
                                             "gpu_before": capture(["nvidia-smi"])})
    record_conditions(root, "resume" if resume else "start")
    total = experiment["rounds"] * len(cells(experiment))
    expected_profile = identity.get("neo_profile_sha256") or {}
    last_round = experiment["rounds"] if until_round is None else min(until_round, experiment["rounds"])
    done, failed, stopped = 0, [], False
    for round_ in range(1, last_round + 1):
        for cell in round_order(experiment, round_):
            done += 1
            path = root / "cells" / f"round-{round_}-{cell['id']}.json"
            if path.exists() and read_json(path).get("status") == "ok":
                continue
            if (root / "STOP").exists():
                stopped = True
                break
            log = root / "logs" / f"round-{round_}-{cell['id']}-{int(time.time())}.log"
            log.parent.mkdir(parents=True, exist_ok=True)
            record = cell_runner(machine, experiment, cell, cases, round_, log)
            atomic_json(path, record)
            if record["status"] != "ok":
                failed.append(path.name)
            profile_now = profile_hashes(Path(machine["neo_profile_dir"]), expected_profile) if expected_profile else {}
            if profile_now != expected_profile:
                atomic_json(root / "status.json", {"phase": "profile-changed", "finished": False,
                                                   "failed_cells": failed + [path.name]})
                raise RuntimeError(f"NEO rewrote its profile during {path.name}; later cells would use other "
                                   "tables. Recalibrate at the top level and start a new run.")
            rates = {language: round(block["generated_tokens_per_s"], 1)
                     for language, block in record["languages"].items()}
            print(f"[{done}/{total}] round {round_} {cell['id']}: {record['status']} {rates} "
                  f"{record.get('error', '')}", flush=True)
        if stopped:
            break
    paused = last_round < experiment["rounds"]
    complete = not stopped and not failed and not paused
    phase = ("complete" if complete else "stopped" if stopped else "incomplete" if failed
             else f"paused after round {last_round}")
    status = {"phase": phase, "finished": complete, "failed_cells": failed}
    atomic_json(root / "status.json", status)
    return status


def top_level(experiment: dict) -> int | None:
    """Largest KV budget any NEO system is started with (None = all free memory)."""
    levels = [b for system in experiment["systems"] if system in NEO_SYSTEMS for b in memory_levels(experiment, system)]
    if not levels:
        raise ValueError("No NEO system in the experiment")
    return None if None in levels else max(levels)


def calibrate_neo(machine_path: str, experiment_path: str, output: str) -> dict:
    """Build NEO's profile once, at the largest budget, then measure its CPU launch time.

    NEO re-profiles and overwrites a table when a start needs a larger one, so the profile must come
    from the largest start: GPU-only gets a few more blocks than the hybrid at the same budget, hence it
    goes first. The hybrid then reuses those tables and measures the launch time (lnch.json).
    """
    machine, experiment = read_json(machine_path), read_json(experiment_path)
    validate_serving(experiment)
    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)
    profile = Path(machine["neo_profile_dir"])
    blocks = top_level(experiment)
    stats = {}
    for system, extra in (("neo_base", ()), ("neo", ("--profile-cpu-launch",))):
        log = root / f"calibrate-{system}-{memory_label(blocks)}-{int(time.time())}.log"
        with serving(system, blocks, machine, experiment, log, extra):
            stats[system] = get_json(machine["port"], "/stats")
    measured = measured_cpu_launch_ms(profile)
    if abs(stats["neo"].get("cpu_launch_time_ms", float("nan")) - measured) > 1e-6:
        raise RuntimeError(f"NEO reported {stats['neo'].get('cpu_launch_time_ms')} ms, lnch.json gives {measured}")
    result = {"machine_id": machine["machine_id"], "top_level": memory_label(blocks),
              "cpu_launch_time_ms": measured, "authors_cpu_launch_ms": AUTHORS_CPU_LAUNCH_MS,
              "lnch": read_json(profile / LAUNCH_PROFILE),
              "profile_sha256": profile_hashes(profile, NEO_PROFILE_FILES + (LAUNCH_PROFILE,)),
              "finished_utc": datetime.now(timezone.utc).isoformat()}
    atomic_json(root / "calibration.json", result)
    print(f"Measured CPU launch time: {measured:.6f} ms per layer (authors: {AUTHORS_CPU_LAUNCH_MS}). "
          f"Set \"neo_cpu_launch_ms\": {measured!r} in {machine_path}.")
    return result


# ------------------------------------------------------------------------- analysis

def load_serving(root: Path) -> tuple[dict, dict, list[dict]]:
    identity = read_json(root / "metadata.json")["fingerprint"]
    experiment, machine = identity["experiment"], identity["machine"]
    rows = []
    for round_ in range(1, experiment["rounds"] + 1):
        for cell in cells(experiment):
            path = root / "cells" / f"round-{round_}-{cell['id']}.json"
            if not path.exists():
                raise ValueError(f"Incomplete run: missing {path.name}")
            record = read_json(path)
            if record["status"] != "ok":
                raise ValueError(f"Failed cell: {path.name}: {record.get('error')}")
            reported = record["kv_reported"]
            requested = record["gpu_blocks_requested"]
            if "error" in reported or (requested is not None and reported["gpu_blocks"] != requested):
                raise ValueError(f"KV capacity mismatch: {path.name}: asked {requested}, reported {reported}")
            for language in experiment["languages"]:
                block = record["languages"][language]
                if block["requests"] != requests_for(cell["load"], experiment):
                    raise ValueError(f"Wrong request count: {path.name} {language}")
                telemetry, work = block["telemetry"], block["work"]
                expected = {"neo": AUTHORS_CPU_LAUNCH_MS, "neo_tuned": machine.get("neo_cpu_launch_ms")}
                if cell["system"] in expected and (work.get("cpu_launch_time_ms") is None or
                                                   abs(work["cpu_launch_time_ms"] - expected[cell["system"]]) > 1e-6):
                    raise ValueError(f"{path.name}: scheduler used CPU launch time {work.get('cpu_launch_time_ms')}, "
                                     f"expected {expected[cell['system']]}")
                rows.append({
                    "system": cell["system"], "memory": memory_label(cell["blocks"]), "load": cell["load"],
                    "language": language, "round": round_, "position": block["position"],
                    "gpu_blocks": reported["gpu_blocks"],
                    **{key: block[key] for key in ("requests", "generated_tokens_per_s",
                                                   "steady_generated_tokens_per_s", "requests_per_s",
                                                   "mean_input_tokens", "mean_latency_s", "p95_latency_s",
                                                   "mean_ttft_s", "mean_time_per_output_token_ms")},
                    "cpu_launch_time_ms": work.get("cpu_launch_time_ms"),
                    "cpu_decode_share": work.get("cpu_decode_share"),
                    "pipelined_share": work.get("pipelined_share"),
                    "swap_out_requests": work.get("swap_out_requests"),
                    "preemptions": work.get("preemptions"),
                    "gpu_util_pct": telemetry["gpu"].get("gpu_util_mean_pct"),
                    "gpu_power_w": telemetry["gpu"].get("power_mean_w"),
                    "cpu_cores": telemetry["process"].get("cpu_cores_mean")})
    return experiment, machine, rows


def aggregate(rows: list[dict]) -> list[dict]:
    """One row per system, memory, load and language: mean over rounds, range of rounds."""
    groups: dict[tuple, list[dict]] = {}
    for row in rows:
        groups.setdefault((row["system"], row["memory"], row["load"], row["language"]), []).append(row)
    table = []
    for (system, memory, load, language), own in groups.items():
        rates = [r["generated_tokens_per_s"] for r in own]
        entry = {"system": system, "memory": memory, "load": load, "language": language, "rounds": len(own),
                 "gpu_blocks": own[0]["gpu_blocks"], "mean_input_tokens": own[0]["mean_input_tokens"],
                 "rate_min": min(rates), "rate_max": max(rates),
                 "round_spread_pct": 100 * (max(rates) - min(rates)) / st.mean(rates)}
        for key in ("generated_tokens_per_s", "steady_generated_tokens_per_s", "requests_per_s", "mean_latency_s",
                    "p95_latency_s", "mean_ttft_s", "mean_time_per_output_token_ms", "cpu_decode_share",
                    "pipelined_share", "swap_out_requests", "preemptions", "gpu_util_pct", "gpu_power_w",
                    "cpu_cores"):
            present = [r[key] for r in own if r[key] is not None]
            entry[key] = st.mean(present) if present else None
        table.append(entry)
    return sorted(table, key=lambda r: (r["memory"], r["language"], r["system"], r["load"]))


def crossovers(table: list[dict], loads: list[int]) -> list[dict]:
    """Per hybrid mode, memory level and language: NEO against the same engine GPU-only at each load.

    NEO wins at a load when its slowest round beats the GPU-only fastest round and GPU-only
    is saturated there: it swapped requests out, or it grew less than its own round range
    since the previous load. The crossover is the smallest load where NEO wins.
    """
    index = {(r["system"], r["memory"], r["load"], r["language"]): r for r in table}
    result = []
    for system, memory, language in sorted({(r["system"], r["memory"], r["language"])
                                            for r in table if r["system"] in HYBRID}):
        steps, crossover, previous = [], None, None
        for load in sorted(loads):
            neo, base = index.get((system, memory, load, language)), index.get(("neo_base", memory, load, language))
            if neo is None or base is None:
                continue
            growth = None if previous is None else base["generated_tokens_per_s"] / previous - 1
            saturated = bool(base["swap_out_requests"]) or (
                growth is not None and 100 * growth < base["round_spread_pct"])
            wins = neo["rate_min"] > base["rate_max"] and saturated
            steps.append({"load": load, "neo": neo["generated_tokens_per_s"], "base": base["generated_tokens_per_s"],
                          "gain_pct": 100 * (neo["generated_tokens_per_s"] / base["generated_tokens_per_s"] - 1),
                          "base_saturated": saturated, "neo_wins": wins})
            if wins and crossover is None:
                crossover = load
            previous = base["generated_tokens_per_s"]
        result.append({"system": system, "memory": memory, "language": language, "crossover_load": crossover,
                       "steps": steps})
    return result


def position_effect(rows: list[dict]) -> list[dict]:
    """Throughput relative to the mean of the same cell and language, by position in the server start."""
    means: dict[tuple, list[float]] = {}
    for row in rows:
        means.setdefault((row["system"], row["memory"], row["load"], row["language"]), []).append(
            row["generated_tokens_per_s"])
    by_position: dict[int, list[float]] = {}
    for row in rows:
        mean = st.mean(means[(row["system"], row["memory"], row["load"], row["language"])])
        by_position.setdefault(row["position"], []).append(row["generated_tokens_per_s"] / mean)
    return [{"position": position, "blocks": len(values), "relative_rate": st.mean(values)}
            for position, values in sorted(by_position.items())]


def report(experiment: dict, machine: dict, table: list[dict], crossing: list[dict], positions: list[dict]) -> str:
    lines = [f"# Serving under load: {experiment['experiment_id']} on {machine['machine_id']}", "",
             f"Closed loop of `load` clients; {experiment['rounds']} rounds, a fresh server start per cell; "
             f"{experiment['output_tokens']} generated tokens per request. Throughput is generated tokens per "
             "second of wall time. NEO is compared with the same engine run with `--always-use-gpu` "
             "(GPU-only) at the same KV-cache budget; vLLM is an external reference. `neo` is NEO as shipped "
             f"(scheduler CPU launch time {AUTHORS_CPU_LAUNCH_MS} ms per layer); `neo_tuned` uses the launch time "
             f"measured on this machine ({machine.get('neo_cpu_launch_ms')} ms).", "",
             "## Where NEO overtakes GPU-only", "",
             "| NEO mode | Memory | Language | Smallest winning load | Gain at each load, % (* = GPU-only saturated) |",
             "|---|---|---|---:|---|"]
    for item in crossing:
        gains = ", ".join(f"{step['load']}: {step['gain_pct']:+.1f}{'*' if step['base_saturated'] else ''}"
                          for step in item["steps"])
        lines.append(f"| {item['system']} | {item['memory']} | {item['language']} | "
                     f"{item['crossover_load'] or 'none'} | {gains} |")
    lines += ["", "## All cells", "",
              "| System | Memory | Load | Language | Input tokens | Generated tok/s | Steady tok/s | Spread, % | "
              "Latency, s | p95, s | TTFT, s | CPU decode share | Swapped out | CPU cores | GPU util, % |",
              "|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]

    def fmt(value, pattern):
        return "n/a" if value is None else format(value, pattern)

    for row in table:
        lines.append(
            f"| {row['system']} | {row['memory']} | {row['load']} | {row['language']} | "
            f"{row['mean_input_tokens']:.1f} | {row['generated_tokens_per_s']:.1f} | "
            f"{fmt(row['steady_generated_tokens_per_s'], '.1f')} | {row['round_spread_pct']:.1f} | "
            f"{row['mean_latency_s']:.2f} | {row['p95_latency_s']:.2f} | {row['mean_ttft_s']:.2f} | "
            f"{fmt(row['cpu_decode_share'], '.2f')} | {fmt(row['swap_out_requests'], '.0f')} | "
            f"{fmt(row['cpu_cores'], '.1f')} | {fmt(row['gpu_util_pct'], '.0f')} |")
    lines += ["", "## Language order within a server start", "",
              "| Position | Blocks | Throughput relative to the cell mean |", "|---:|---:|---:|"]
    lines += [f"| {p['position']} | {p['blocks']} | {p['relative_rate']:.3f} |" for p in positions]
    return "\n".join(lines) + "\n"


def analyze_serving(root: str | Path, output: str | Path) -> None:
    root, output = Path(root), Path(output)
    experiment, machine, rows = load_serving(root)
    table = aggregate(rows)
    crossing = crossovers(table, experiment["loads"])
    positions = position_effect(rows)
    output.mkdir(parents=True, exist_ok=True)
    for name, data in (("rounds.csv", rows), ("cells.csv", table)):
        with (output / name).open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(data[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(data)
    (output / "crossovers.json").write_text(json.dumps(crossing, indent=2) + "\n", encoding="utf-8")
    (output / "report.md").write_text(report(experiment, machine, table, crossing, positions), encoding="utf-8")
    print(f"Validated {len(rows)} blocks. Report: {output / 'report.md'}")
