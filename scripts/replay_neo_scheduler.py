"""Replay NEO's offload decision on a machine's profile tables, without a GPU.

NEO (swiftllm/server/scheduler.py, _decide_mode_and_gen_batch) moves a request's decoding to
the CPU only if, for both pipelined sub-batches, the GPU time of the other sub-batch covers this
one's CPU time: linr_T(other) + pref_T + gdec_T - (cdec_T + lnch_T) >= 0, per layer, in ms. It
then keeps the pipelined pair only if its predicted token rate beats the GPU-only batch. This
script repeats that arithmetic on the tables NEO profiled (linr, gdec, cdec JSON), for a
decode-only iteration: G requests decoding on the GPU and C waiting in the CPU queue, all of
sequence length L. It shows where the scheduler would use the CPU at a given launch time.

    python scripts/replay_neo_scheduler.py PROFILE_DIR [PROFILE_DIR ...]

Each PROFILE_DIR is a NEO --profile-result-path directory. Predictions only: whether the CPU
really pays off is measured by the serving runner.
"""

import argparse
import bisect
import json
from pathlib import Path

LAUNCH_TIMES_MS = (0.8, 0.4, 0.2, 0.1, 0.0)  # 0.8 is the value hard-coded by NEO's authors
BLOCK_SIZE = 16
LANGUAGES = {"eng_Latn": 109 + 64, "kaz_Cyrl": 284 + 64}  # mean prompt (Llama-2 tokens) + half of 128 output
LOAD = 64


def interpolate(x: float, xs: list[float], ys: list[float]) -> float:
    """NEO's TablePerfPredictor._interp_1d: linear between profiled points, exact at them."""
    if x == 0:
        return 0.0
    if x > xs[-1]:
        raise ValueError(f"{x} beyond the profiled range {xs[-1]}")
    i = bisect.bisect_left(xs, x)
    if i == 0 or xs[i] == x:
        return ys[i]
    return ys[i - 1] + (ys[i] - ys[i - 1]) * (x - xs[i - 1]) / (xs[i] - xs[i - 1])


class Tables:
    def __init__(self, directory: Path):
        def load(name):
            return json.loads((directory / f"{name}.json").read_text())

        self.linr, self.gdec, cdec = load("linr"), load("gdec"), load("cdec")
        self.cdec_sizes = cdec["S_list"]
        self.cdec_kv = sorted({n for row in cdec["N_lists"] for n in row})
        self.cdec_times = cdec["T_lists"]

    def linr_t(self, tokens: int) -> float:
        return interpolate(tokens, self.linr["S_list"], self.linr["T_list"])

    def gdec_t(self, kv_tokens: int) -> float:
        # Tables profiled at a smaller budget end early; beyond them the last point is used.
        return interpolate(min(kv_tokens, self.gdec["N_list"][-1]), self.gdec["N_list"], self.gdec["T_list"])

    def cdec_t(self, requests: int, kv_tokens: int) -> float:
        if requests == 0:
            return 0.0
        i = bisect.bisect_left(self.cdec_sizes, requests)
        upper = interpolate(kv_tokens, self.cdec_kv, self.cdec_times[i])
        if self.cdec_sizes[i] == requests:
            return upper
        lower = interpolate(kv_tokens, self.cdec_kv, self.cdec_times[i - 1])
        return lower + (upper - lower) * (requests - self.cdec_sizes[i - 1]) / (self.cdec_sizes[i] - self.cdec_sizes[i - 1])


class SubBatch:
    def __init__(self, tables: Tables, launch_ms: float):
        self.tables, self.launch_ms = tables, launch_ms
        self.tokens = self.gpu_kv = self.cpu_requests = self.cpu_kv = self.requests = 0

    def gpu_time(self) -> float:
        return self.tables.linr_t(self.tokens) + self.tables.gdec_t(self.gpu_kv)

    def cpu_time(self) -> float:
        return self.tables.cdec_t(self.cpu_requests, self.cpu_kv) + self.launch_ms

    def move_cpu(self, length: int, sign: int) -> None:
        self.tokens += sign
        self.requests += sign
        self.cpu_requests += sign
        self.cpu_kv += sign * length


def decide(tables: Tables, launch_ms: float, gpu: int, cpu: int, length: int) -> tuple[int, float]:
    """(requests the scheduler puts on the CPU, predicted pipelined/GPU-only rate), (0, 0) if none."""
    batches = [SubBatch(tables, launch_ms), SubBatch(tables, launch_ms)]
    gpu_only = SubBatch(tables, launch_ms)
    for batch in (batches[0], gpu_only):
        batch.tokens, batch.gpu_kv, batch.requests = gpu, gpu * length, gpu

    def remains() -> list[float]:
        return [batches[j ^ 1].tables.linr_t(batches[j ^ 1].tokens) + batches[j].tables.gdec_t(batches[j].gpu_kv)
                - batches[j].cpu_time() for j in range(2)]

    target, accepted = 1, 0
    for _ in range(cpu):  # all requests have the same length: the first refusal ends the scan, as in NEO
        batches[target].move_cpu(length, +1)
        slack = remains()
        if min(slack) < 0:
            batches[target].move_cpu(length, -1)
            break
        accepted += 1
        target = int(slack[1] > slack[0])
    if not batches[1].requests:
        return 0, 0.0
    sequential = gpu_only.requests / gpu_only.gpu_time()
    pipelined = (batches[0].requests + batches[1].requests) / (batches[0].gpu_time() + batches[1].gpu_time())
    return (accepted, pipelined / sequential) if pipelined > sequential else (0, 0.0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("profiles", nargs="+", type=Path)
    parser.add_argument("--blocks", type=int, nargs="+", default=[64, 96, 128, 192, 256])
    args = parser.parse_args()
    for directory in args.profiles:
        tables = Tables(directory)
        print(f"## {directory}")
        print(f"linear part per layer: 1 token {tables.linr_t(1):.3f} ms, 64 tokens {tables.linr_t(64):.3f} ms; "
              f"CPU attention for one {LANGUAGES['kaz_Cyrl']}-token request {tables.cdec_t(1, 348):.3f} ms\n")
        header = " | ".join(f"lnch {t}" for t in LAUNCH_TIMES_MS)
        print(f"| KV blocks | Language | GPU requests | {header} |")
        print("|---:|---|---:|" + "---|" * len(LAUNCH_TIMES_MS))
        for blocks in args.blocks:
            for language, length in LANGUAGES.items():
                gpu = min(LOAD, blocks * BLOCK_SIZE // length)
                cells = []
                for launch in LAUNCH_TIMES_MS:
                    accepted, ratio = decide(tables, launch, gpu, LOAD - gpu, length)
                    cells.append(f"{accepted} on CPU, x{ratio:.2f}" if accepted else "GPU only")
                print(f"| {blocks} | {language} | {gpu} | " + " | ".join(cells) + " |")
        print()


if __name__ == "__main__":
    main()
