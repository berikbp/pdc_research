"""Per-request GPU and server-process telemetry, plus run-condition snapshots.

GPU readings come from one streaming nvidia-smi process per server load. Server CPU
time and memory are read from /proc exactly at request boundaries. A missing tool or
process is stored as an error; it must never look like a zero reading.
"""

import os
import subprocess
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from .common import capture

GPU_QUERY = "memory.used,utilization.gpu,power.draw,temperature.gpu"
GPU_KEYS = ("vram_mib", "gpu_util_pct", "power_w", "temp_c")
INTERVAL_MS = 100
FIRST_SAMPLE_TIMEOUT_SECONDS = 5
NO_SERVER = "no server process recorded; telemetry skipped"


def pid_file(log: str | Path) -> Path:
    return Path(str(log) + ".pid")


def parse_gpu_line(line: str) -> dict | None:
    parts = [part.strip() for part in line.split(",")]
    if len(parts) != len(GPU_KEYS):
        return None
    try:
        return dict(zip(GPU_KEYS, map(float, parts)))
    except ValueError:
        return None  # e.g. "[N/A]" for a field this GPU or driver cannot report


def summarize_gpu(samples: list[tuple[float, dict]], start: float, end: float) -> dict:
    window = [reading for moment, reading in samples if start <= moment <= end]
    carried = False
    if not window:
        earlier = [reading for moment, reading in samples if moment < start]
        if not earlier:
            return {"samples": 0, "error": "no GPU samples collected"}
        window, carried = [earlier[-1]], True
    power = sum(reading["power_w"] for reading in window) / len(window)
    summary = {
        "samples": 0 if carried else len(window),
        "vram_max_mib": max(reading["vram_mib"] for reading in window),
        "gpu_util_mean_pct": sum(reading["gpu_util_pct"] for reading in window) / len(window),
        "gpu_util_max_pct": max(reading["gpu_util_pct"] for reading in window),
        "power_mean_w": power,
        "energy_j": power * (end - start),
        "temp_max_c": max(reading["temp_c"] for reading in window),
    }
    return {**summary, "carried_from_earlier_sample": True} if carried else summary


def read_process(pid: int) -> dict:
    """CPU seconds over all threads and resident memory of one process."""
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        status = dict(line.split(":", 1) for line in Path(f"/proc/{pid}/status").read_text().splitlines()
                      if ":" in line)
        return {
            "cpu_seconds": (int(fields[11]) + int(fields[12])) / os.sysconf("SC_CLK_TCK"),
            "rss_mib": int(status["VmRSS"].split()[0]) / 1024,
            "peak_rss_mib": int(status["VmHWM"].split()[0]) / 1024,
            "threads": int(fields[17]),
        }
    except (OSError, KeyError, IndexError, ValueError) as exc:
        return {"error": f"cannot read process {pid}: {exc}"}


def process_tree(pid: int) -> list[int]:
    """The process and all its live descendants (vLLM computes in a child process)."""
    children: dict[int, list[int]] = {}
    for entry in Path("/proc").iterdir():
        if entry.name.isdigit():
            try:
                parent = int((entry / "stat").read_text().rsplit(")", 1)[1].split()[1])
            except (OSError, IndexError, ValueError):
                continue
            children.setdefault(parent, []).append(int(entry.name))
    tree, stack = [], [pid]
    while stack:
        current = stack.pop()
        tree.append(current)
        stack.extend(children.get(current, []))
    return tree


def read_process_tree(pid: int) -> dict:
    """CPU seconds, memory and threads summed over the process and its descendants.

    CPU time of descendants that exit inside a window is not counted; the servers
    measured here keep their workers alive for their whole lifetime.
    """
    root = read_process(pid)
    if "error" in root:
        return root
    readings = [root] + [reading for child in process_tree(pid)[1:]
                         if "error" not in (reading := read_process(child))]
    total = {key: sum(reading[key] for reading in readings)
             for key in ("cpu_seconds", "rss_mib", "peak_rss_mib", "threads")}
    return {**total, "processes": len(readings)}


def process_delta(before: dict, after: dict, seconds: float) -> dict:
    for reading in (before, after):
        if "error" in reading:
            return {"error": reading["error"]}
    cpu = after["cpu_seconds"] - before["cpu_seconds"]
    return {"cpu_seconds": cpu, "cpu_cores_mean": cpu / seconds if seconds > 0 else None,
            "rss_mib": after["rss_mib"], "peak_rss_mib": after["peak_rss_mib"],
            "threads": after["threads"], **({"processes": after["processes"]} if "processes" in after else {})}


def server_gpu_memory(pid: int, pids: list[int] | None = None) -> dict:
    """VRAM attributed to the server itself (and `pids`, its workers), excluding the desktop."""
    owners = {str(item) for item in (pids or [pid])}
    result = capture(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"])
    if result.get("returncode") != 0:
        return {"error": result.get("error") or result.get("stderr", "").strip() or "nvidia-smi failed"}
    total = 0
    for line in result["stdout"].splitlines():
        owner, _, used = line.partition(",")
        if owner.strip() in owners and used.strip().isdigit():
            total += int(used)
    return {"vram_mib": total}


class Sampler:
    def __init__(self, pid: int | None, interval_ms: int = INTERVAL_MS, *,
                 gpu_index: int | None = None, tree: bool = False):
        """`gpu_index` restricts sampling to one GPU; `tree` counts the server's child processes."""
        self.pid = pid
        self.gpu_index = gpu_index
        self.tree = tree
        self.interval_ms = interval_ms
        self.samples: list[tuple[float, dict]] = []
        self.gpu_error: str | None = None
        self.server_gpu: dict | None = None
        self._process: subprocess.Popen | None = None
        self._first = threading.Event()

    def _read(self) -> None:
        for line in self._process.stdout:
            reading = parse_gpu_line(line)
            if reading is None:
                self.gpu_error = f"unparsed nvidia-smi output: {line.strip()!r}"
                continue
            self.samples.append((time.perf_counter(), reading))
            self._first.set()

    def __enter__(self) -> "Sampler":
        if self.pid is None:
            return self
        try:
            self._process = subprocess.Popen(
                ["nvidia-smi", *(["-i", str(self.gpu_index)] if self.gpu_index is not None else []),
                 f"--query-gpu={GPU_QUERY}", "--format=csv,noheader,nounits", "-lms", str(self.interval_ms)],
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
        except OSError as exc:
            self.gpu_error = f"nvidia-smi unavailable: {exc}"
            return self
        threading.Thread(target=self._read, daemon=True).start()
        if not self._first.wait(FIRST_SAMPLE_TIMEOUT_SECONDS):
            self.gpu_error = self.gpu_error or "nvidia-smi produced no samples"
        self.server_gpu = server_gpu_memory(self.pid, process_tree(self.pid) if self.tree else None)
        return self

    def __exit__(self, *_) -> None:
        if self._process is not None and self._process.poll() is None:
            self._process.terminate()
            try:
                self._process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait()

    def mark(self) -> dict:
        if self.pid is None:
            return {}
        return {"time": time.perf_counter(), "process": self._read_process()}

    def _read_process(self) -> dict:
        return read_process_tree(self.pid) if self.tree else read_process(self.pid)

    def since(self, mark: dict) -> dict:
        if self.pid is None:
            return {"error": NO_SERVER}
        end = time.perf_counter()
        seconds = end - mark["time"]
        if self.samples:
            gpu = summarize_gpu(list(self.samples), mark["time"], end)
        else:
            gpu = {"samples": 0, "error": self.gpu_error or "no GPU samples collected"}
        return {"window_seconds": seconds, "interval_ms": self.interval_ms, "gpu": gpu,
                "server_gpu_memory": self.server_gpu,
                "process": process_delta(mark["process"], self._read_process(), seconds)}


@contextmanager
def sampler(log: str | Path) -> Iterator[Sampler]:
    """Sample the server whose PID backend.server stored beside this log."""
    try:
        pid = int(pid_file(log).read_text())
    except (OSError, ValueError):
        pid = None
    with Sampler(pid) as probe:
        yield probe


def _read_text(path: str) -> str | None:
    try:
        return Path(path).read_text().strip()
    except OSError:
        return None


def run_conditions() -> dict:
    """Power and load state at start or resume; values that cannot be read are null."""
    supplies = {path.name: _read_text(str(path / "online"))
                for path in sorted(Path("/sys/class/power_supply").glob("*"))
                if (path / "online").exists()} if Path("/sys/class/power_supply").is_dir() else {}
    return {
        "recorded_utc": datetime.now(timezone.utc).isoformat(),
        "cpu_governor": _read_text("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor"),
        "cpu_energy_preference": _read_text("/sys/devices/system/cpu/cpu0/cpufreq/energy_performance_preference"),
        "platform_profile": _read_text("/sys/firmware/acpi/platform_profile"),
        "power_supplies_online": supplies,
        "load_average": _read_text("/proc/loadavg"),
        "memory": capture(["free", "-m"]),
        "gpu": capture(["nvidia-smi", f"--query-gpu={GPU_QUERY},memory.total,driver_version",
                        "--format=csv,noheader,nounits"]),
        "gpu_processes": capture(["nvidia-smi", "--query-compute-apps=pid,name,used_memory",
                                  "--format=csv,noheader,nounits"]),
    }
