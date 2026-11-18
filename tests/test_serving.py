import json
import os
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from contextlib import contextmanager
from pathlib import Path

from pdc_research import serving
from pdc_research.serving import (analyze_serving, calibrate_neo, cells, counter_delta, crossovers, drive,
                                  fingerprint, language_order, load_serving, measured_cpu_launch_ms, parse_stream,
                                  reported_kv, requests_for, round_order, run_serving, server_command,
                                  server_environment, summarize_requests, top_level, validate_serving)
from pdc_research.telemetry import read_process_tree

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = json.loads((ROOT / "configs/experiments/serving-llama2-pc1.json").read_text())
EXPERIMENT_PC2 = json.loads((ROOT / "configs/experiments/serving-llama2-pc2.json").read_text())
MACHINE = {**json.loads((ROOT / "configs/machines/lab-pc1.json").read_text()), "neo_cpu_launch_ms": 0.15}
LAUNCH = {"neo": 0.8, "neo_tuned": 0.15}
SMALL = {**EXPERIMENT, "systems": ["neo", "neo_base"], "memory_blocks": [64], "loads": [1, 4],
         "languages": ["eng_Latn", "kaz_Cyrl"], "rounds": 2}
CASES = {"tokenizer": {"tokenizer_json_sha256": "x"},
         "cases": {language: [{"group_id": f"dev-{i}", "tokens": [1] + [5] * (50 + i)} for i in range(20)]
                   for language in ("eng_Latn", "kaz_Cyrl")}}


def block(rate: float, position: int = 0, swapped: int = 0, launch: float | None = None) -> dict:
    return {"position": position, "requests": 32, "generated_tokens_per_s": rate,
            "steady_generated_tokens_per_s": rate, "requests_per_s": rate / 128, "mean_input_tokens": 100.0,
            "mean_latency_s": 1.0, "p95_latency_s": 2.0, "mean_ttft_s": 0.1, "mean_time_per_output_token_ms": 8.0,
            "work": {"cpu_decode_share": 0.0, "pipelined_share": 0.0, "swap_out_requests": swapped,
                     "cpu_launch_time_ms": launch},
            "telemetry": {"gpu": {"gpu_util_mean_pct": 90.0, "power_mean_w": 300.0},
                          "process": {"cpu_cores_mean": 1.0}}}


def write_run(root: Path, rates: dict, reported: int | None = None, experiment: dict = SMALL,
              launch: dict = LAUNCH) -> None:
    """rates[(system, load)] = (round 1, round 2) throughput, same for both languages."""
    (root / "cells").mkdir(parents=True)
    (root / "metadata.json").write_text(json.dumps({"fingerprint": {"experiment": experiment, "machine": MACHINE}}))
    for cell in cells(experiment):
        requested = serving.gpu_blocks_requested(cell["system"], cell["blocks"], MACHINE)
        for round_ in (1, 2):
            rate = rates[(cell["system"], cell["load"])][round_ - 1]
            swapped = 5 if cell["system"] == "neo_base" and cell["load"] == 4 else 0
            record = {"status": "ok", "cell": cell, "gpu_blocks_requested": requested,
                      "kv_reported": {"gpu_blocks": requested if reported is None else reported},
                      "languages": {language: block(rate, position, swapped, launch.get(cell["system"]))
                                    for position, language in enumerate(experiment["languages"])}}
            (root / "cells" / f"round-{round_}-{cell['id']}.json").write_text(json.dumps(record))


class PlanTests(unittest.TestCase):
    def test_shipped_experiment_is_valid_and_fits_its_workload(self):
        validate_serving(EXPERIMENT)
        validate_serving(EXPERIMENT_PC2)
        # PC1: GPU-only and tuned NEO at five budgets, NEO as shipped only at 64 blocks; six loads each.
        self.assertEqual(len(cells(EXPERIMENT)), (5 + 5 + 1) * 6)
        # PC2: two budgets for NEO, vLLM also with all free memory.
        self.assertEqual(len(cells(EXPERIMENT_PC2)), (2 + 2 + 1 + 3) * 6)
        self.assertEqual({c["blocks"] for c in cells(EXPERIMENT) if c["system"] == "neo"}, {64})
        self.assertEqual((top_level(EXPERIMENT), top_level(EXPERIMENT_PC2)), (None, 96))

    def test_per_system_budgets_must_name_measured_systems(self):
        with self.assertRaisesRegex(ValueError, "not in the experiment"):
            validate_serving({**EXPERIMENT, "memory_blocks_by_system": {"vllm": [64]}})
        with self.assertRaisesRegex(ValueError, "max_tokens_in_batch"):
            validate_serving({**EXPERIMENT, "memory_blocks_by_system": {"neo": [60]}})

    def test_budget_too_small_for_a_prefill_batch_is_refused(self):
        with self.assertRaisesRegex(ValueError, "max_tokens_in_batch"):
            validate_serving({**EXPERIMENT, "memory_blocks": [60]})
        with self.assertRaisesRegex(ValueError, "largest load"):
            validate_serving({**EXPERIMENT, "loads": [1, 512]})

    def test_requests_are_whole_waves_with_a_floor(self):
        self.assertEqual([requests_for(load, EXPERIMENT) for load in (1, 4, 8, 16, 32, 64, 24)],
                         [32, 32, 32, 64, 128, 256, 96])

    def test_orders_are_seeded_and_change_between_rounds(self):
        self.assertEqual(round_order(EXPERIMENT, 1), round_order(EXPERIMENT, 1))
        self.assertNotEqual(round_order(EXPERIMENT, 1), round_order(EXPERIMENT, 2))
        first = language_order(EXPERIMENT, 1, "neo-b0064-c16")
        self.assertEqual(sorted(first), sorted(EXPERIMENT["languages"]))
        self.assertEqual(first, language_order(EXPERIMENT, 1, "neo-b0064-c16"))


class ServerTests(unittest.TestCase):
    def test_commands_split_memory_as_neo_does(self):
        neo = server_command("neo", 64, MACHINE, EXPERIMENT)
        base = server_command("neo_base", 64, MACHINE, EXPERIMENT)
        vllm = server_command("vllm", 64, MACHINE, EXPERIMENT)
        self.assertEqual(neo[neo.index("--num-gpu-blocks-override") + 1], "62")
        self.assertIn("--extra-layer-for-cprf", neo)
        self.assertEqual(base[base.index("--num-gpu-blocks-override") + 1], "64")
        self.assertIn("--always-use-gpu", base)
        self.assertEqual(neo[neo.index("--swap-space") + 1], base[base.index("--swap-space") + 1])
        self.assertEqual(vllm[vllm.index("--num-gpu-blocks-override") + 1], "64")
        self.assertIn("--no-enable-prefix-caching", vllm)
        self.assertNotIn("--num-gpu-blocks-override", server_command("neo", None, MACHINE, EXPERIMENT))
        # NEO profiles a single prefill of max_tokens_in_batch (960) tokens: 60 blocks, not 768 / 16 = 48.
        self.assertEqual(neo[neo.index("--max-blocks-per-seq") + 1], "60")

    def test_tuned_neo_differs_from_shipped_neo_only_in_launch_time(self):
        neo = server_command("neo", 64, MACHINE, EXPERIMENT)
        tuned = server_command("neo_tuned", 64, MACHINE, EXPERIMENT)
        self.assertNotIn("--cpu-launch-time", neo)  # as shipped: NEO's own default, the authors' 0.8 ms
        self.assertEqual(tuned[tuned.index("--cpu-launch-time") + 1], "0.15")
        without = [arg for arg in tuned if arg not in ("--cpu-launch-time", "0.15")]
        self.assertEqual(without, neo)
        with self.assertRaisesRegex(ValueError, "calibrate-neo"):
            server_command("neo_tuned", 64, {**MACHINE, "neo_cpu_launch_ms": None}, EXPERIMENT)

    def test_environment_selects_one_gpu_and_binds_only_neo_threads(self):
        machine = {**MACHINE, "omp_threads": 8, "omp_cpu_affinity": "0-7"}
        neo, vllm = server_environment("neo", machine), server_environment("vllm", machine)
        self.assertEqual((neo["CUDA_VISIBLE_DEVICES"], neo["CUDA_DEVICE_ORDER"]), ("0", "PCI_BUS_ID"))
        self.assertEqual((neo["OMP_NUM_THREADS"], neo["GOMP_CPU_AFFINITY"]), ("8", "0-7"))
        self.assertNotIn("GOMP_CPU_AFFINITY", {k: v for k, v in vllm.items() if k not in os.environ})

    def test_reported_kv_capacity_is_read_from_the_log(self):
        neo_log = "INFO [Engine.profiler] Number of GPU blocks: 62 (0.48 GB)\n"
        self.assertEqual(reported_kv("neo", neo_log, 16)["gpu_blocks"], 62)
        self.assertEqual(reported_kv("vllm", "INFO GPU KV cache size: 1,024 tokens\n", 16)["gpu_blocks"], 64)
        self.assertIn("error", reported_kv("neo_base", "nothing here", 16))


class ClientTests(unittest.TestCase):
    def test_neo_stream_counts_token_lines(self):
        clock = iter(range(10, 100))
        parsed = parse_stream("neo", iter([b"5\n", b"\n", b"7\n", b"9\n"]), lambda: next(clock))
        self.assertEqual((parsed["output_tokens"], parsed["first"]), (3, 10))

    def test_vllm_stream_prefers_reported_usage(self):
        lines = [b'data: {"choices":[{"text":"a"}]}\n', b'data: {"choices":[{"text":"bc"}]}\n',
                 b'data: {"choices":[],"usage":{"completion_tokens":3}}\n', b"data: [DONE]\n"]
        parsed = parse_stream("vllm", iter(lines), time.perf_counter)
        self.assertEqual(parsed["output_tokens"], 3)

    def test_summary_rejects_wrong_lengths_and_measures_wall_time(self):
        requests = [{"started_s": start, "ttft_s": 0.2, "elapsed_s": 2.0, "output_tokens": 128,
                     "input_tokens": 100} for start in (0.0, 0.0, 2.0, 2.0)]
        summary = summarize_requests(requests, 128)
        self.assertAlmostEqual(summary["wall_s"], 4.0)
        self.assertAlmostEqual(summary["generated_tokens_per_s"], 128.0)
        self.assertAlmostEqual(summary["mean_time_per_output_token_ms"], 1000 * 1.8 / 127)
        with self.assertRaisesRegex(ValueError, "exactly 128"):
            summarize_requests([{**requests[0], "output_tokens": 127}], 128)

    def test_closed_loop_never_exceeds_the_load(self):
        active, peak, lock = [0], [0], threading.Lock()

        def send(tokens):
            with lock:
                active[0] += 1
                peak[0] = max(peak[0], active[0])
            started = time.perf_counter()
            time.sleep(0.01)
            with lock:
                active[0] -= 1
            return {"started": started, "first": started, "ended": time.perf_counter(), "output_tokens": 1,
                    "output_sha256": ""}

        results = drive(send, [{"group_id": "g", "tokens": [1, 2]}] * 12, 4)
        self.assertEqual((len(results), peak[0]), (12, 4))

    def test_counter_delta_derives_cpu_share(self):
        before = {"iterations": 10, "pipelined_iterations": 0, "gpu_decode_tokens": 100, "cpu_decode_tokens": 0,
                  "cpu_queue_max": 3}
        after = {"iterations": 20, "pipelined_iterations": 5, "gpu_decode_tokens": 160, "cpu_decode_tokens": 20,
                 "cpu_queue_max": 7}
        delta = counter_delta(before, after)
        self.assertAlmostEqual(delta["cpu_decode_share"], 0.25)
        self.assertAlmostEqual(delta["pipelined_share"], 0.5)
        self.assertEqual(delta["cpu_queue_max_since_start"], 7)
        launch = counter_delta({**before, "cpu_launch_time_ms": 0.15}, {**after, "cpu_launch_time_ms": 0.15})
        self.assertEqual(launch["cpu_launch_time_ms"], 0.15)  # a setting, not a difference of zero
        self.assertIn("error", counter_delta({"error": "down"}, after))


class RunTests(unittest.TestCase):
    def run_with(self, directory: str, outcome, **kwargs) -> tuple[dict, Path]:
        base = Path(directory)
        (base / "m.json").write_text(json.dumps(MACHINE))
        (base / "e.json").write_text(json.dumps(SMALL))
        (base / "c.json").write_text(json.dumps(CASES))
        calls = []

        def fake(machine, experiment, cell, cases, round_, log):
            calls.append((round_, cell["id"]))
            return {"status": outcome(cell, round_), "error": "boom", "languages": {}}

        status = run_serving(str(base / "m.json"), str(base / "e.json"), str(base / "c.json"),
                             str(base / "run"), cell_runner=fake, identity={"id": 1}, **kwargs)
        return status, base / "run", calls

    def test_failed_cell_leaves_run_unfinished_and_resume_retries_only_it(self):
        with tempfile.TemporaryDirectory() as directory:
            failing = {"cell": None}

            def first(cell, round_):
                if failing["cell"] is None:
                    failing["cell"] = (round_, cell["id"])
                return "failed" if (round_, cell["id"]) == failing["cell"] else "ok"

            status, root, calls = self.run_with(directory, first)
            self.assertFalse(status["finished"])
            self.assertEqual(len(calls), 2 * len(cells(SMALL)))
            status, _, calls = self.run_with(directory, lambda cell, round_: "ok", resume=True)
            self.assertTrue(status["finished"])
            self.assertEqual(calls, [failing["cell"]])

    def test_stop_file_created_during_a_run_stops_after_the_current_cell(self):
        with tempfile.TemporaryDirectory() as directory:
            def stop_after_first(cell, round_):
                (Path(directory) / "run" / "STOP").touch()
                return "ok"

            status, _, calls = self.run_with(directory, stop_after_first)
            self.assertEqual((status["phase"], status["finished"], len(calls)), ("stopped", False, 1))

    def test_stop_file_pauses_and_changed_identity_refuses_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            (base / "run").mkdir()
            (base / "run" / "metadata.json").write_text(json.dumps({"fingerprint": {"id": 1}}))
            (base / "run" / "STOP").touch()
            status, root, calls = self.run_with(directory, lambda cell, round_: "ok", resume=True)
            self.assertTrue(status["finished"])  # explicit resume acknowledges an earlier STOP
            (root / "metadata.json").write_text(json.dumps({"fingerprint": {"id": 2}}))
            with self.assertRaisesRegex(ValueError, "Resume refused"):
                self.run_with(directory, lambda cell, round_: "ok", resume=True)


    def test_until_round_pauses_and_resume_finishes(self):
        with tempfile.TemporaryDirectory() as directory:
            status, _, calls = self.run_with(directory, lambda cell, round_: "ok", until_round=1)
            self.assertEqual((status["phase"], status["finished"]), ("paused after round 1", False))
            self.assertEqual({round_ for round_, _ in calls}, {1})
            status, _, calls = self.run_with(directory, lambda cell, round_: "ok", resume=True)
            self.assertTrue(status["finished"])
            self.assertEqual({round_ for round_, _ in calls}, {2})

    def test_profile_rewritten_during_a_cell_stops_the_run(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            profile = base / "profile"
            profile.mkdir()
            for name in serving.NEO_PROFILE_FILES:
                (profile / name).write_text("{}")
            machine = {**MACHINE, "neo_profile_dir": str(profile)}
            for name, data in (("m.json", machine), ("e.json", SMALL), ("c.json", CASES)):
                (base / name).write_text(json.dumps(data))
            identity = {"neo_profile_sha256": serving.profile_hashes(profile, serving.NEO_PROFILE_FILES)}

            def rewrites(machine, experiment, cell, cases, round_, log):
                (profile / "gdec.json").write_text('{"N_list": [1]}')
                return {"status": "ok", "languages": {}}

            with self.assertRaisesRegex(RuntimeError, "rewrote its profile"):
                run_serving(str(base / "m.json"), str(base / "e.json"), str(base / "c.json"), str(base / "run"),
                            cell_runner=rewrites, identity=identity)
            status = json.loads((base / "run" / "status.json").read_text())
            self.assertEqual((status["phase"], len(status["failed_cells"])), ("profile-changed", 1))


class CalibrationTests(unittest.TestCase):
    def fixture(self, directory: str, launch: float | None) -> tuple[dict, dict, str]:
        base = Path(directory)
        model, profile = base / "model", base / "profile"
        model.mkdir()
        profile.mkdir()
        (model / "tokenizer.json").write_text("tok")
        (model / "config.json").write_text("{}")
        (model / "w.safetensors").write_text("w")
        (base / "lib.so").write_text("lib")
        for name in serving.NEO_PROFILE_FILES:
            (profile / name).write_text("{}")
        (profile / "lnch.json").write_text(json.dumps({"S_list": [20, 40], "T_list": [[0.1, 0.2], [0.3, 0.4]]}))
        cases = {**CASES, "tokenizer": {"tokenizer_json_sha256": serving.sha256(model / "tokenizer.json")}}
        (base / "cases.json").write_text(json.dumps(cases))
        machine = {**MACHINE, "model": str(model), "neo_profile_dir": str(profile), "neo_library": str(base / "lib.so"),
                   "neo_dir": str(base), "python": sys.executable, "neo_cpu_launch_ms": launch}
        return machine, {**SMALL, "systems": ["neo_base", "neo_tuned"]}, str(base / "cases.json")

    def test_launch_time_is_the_mean_neo_itself_computes(self):
        with tempfile.TemporaryDirectory() as directory:
            machine, _, _ = self.fixture(directory, 0.25)
            self.assertAlmostEqual(measured_cpu_launch_ms(Path(machine["neo_profile_dir"])), 0.25)

    def test_tuned_neo_must_use_the_measured_value(self):
        with tempfile.TemporaryDirectory() as directory:
            machine, experiment, cases = self.fixture(directory, 0.25)
            identity = fingerprint(machine, experiment, cases)
            self.assertIn("lnch.json", identity["neo_profile_sha256"])
            with self.assertRaisesRegex(ValueError, "measured by calibrate-neo"):
                fingerprint({**machine, "neo_cpu_launch_ms": 0.1}, experiment, cases)

    def test_calibration_builds_the_profile_at_the_top_budget_gpu_only_first(self):
        with tempfile.TemporaryDirectory() as directory:
            machine, experiment, _ = self.fixture(directory, None)
            experiment = {**experiment, "systems": ["neo_base", "neo_tuned", "neo"],
                          "memory_blocks": [64, 128], "memory_blocks_by_system": {"neo": [64]}}
            base = Path(directory)
            for name, data in (("m.json", machine), ("e.json", experiment)):
                (base / name).write_text(json.dumps(data))
            starts = []

            @contextmanager
            def fake_serving(system, blocks, machine, experiment, log, extra=()):
                starts.append((system, blocks, tuple(extra)))
                yield 1

            original = serving.serving, serving.get_json
            serving.serving = fake_serving
            serving.get_json = lambda port, path, timeout=5: {"cpu_launch_time_ms": 0.25}
            try:
                result = calibrate_neo(str(base / "m.json"), str(base / "e.json"), str(base / "cal"))
            finally:
                serving.serving, serving.get_json = original
            self.assertEqual(starts, [("neo_base", 128, ()), ("neo", 128, ("--profile-cpu-launch",))])
            self.assertAlmostEqual(result["cpu_launch_time_ms"], 0.25)
            self.assertTrue((base / "cal" / "calibration.json").is_file())


class AnalysisTests(unittest.TestCase):
    RATES = {("neo", 1): (100.0, 102.0), ("neo_base", 1): (101.0, 103.0),
             ("neo", 4): (300.0, 310.0), ("neo_base", 4): (200.0, 204.0)}

    def test_crossover_needs_a_clear_win_over_a_saturated_baseline(self):
        with tempfile.TemporaryDirectory() as directory:
            root, output = Path(directory) / "run", Path(directory) / "out"
            write_run(root, self.RATES)
            analyze_serving(root, output)
            crossing = json.loads((output / "crossovers.json").read_text())
            self.assertEqual({item["crossover_load"] for item in crossing}, {4})
            self.assertFalse(crossing[0]["steps"][0]["neo_wins"])
            self.assertIn("Where NEO overtakes GPU-only", (output / "report.md").read_text())

    def test_tuned_neo_is_compared_with_gpu_only_on_its_own(self):
        tuned = {**SMALL, "systems": ["neo", "neo_tuned", "neo_base"]}
        # Tuned NEO at load 4 overlaps GPU-only (200-204): no win, although NEO as shipped wins there.
        rates = {**self.RATES, ("neo_tuned", 1): (100.0, 101.0), ("neo_tuned", 4): (202.0, 206.0)}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "run"
            write_run(root, rates, experiment=tuned)
            experiment, machine, rows = load_serving(root)
            crossing = {item["system"]: item for item in crossovers(serving.aggregate(rows), experiment["loads"])
                        if item["language"] == "eng_Latn"}
            self.assertEqual((crossing["neo"]["crossover_load"], crossing["neo_tuned"]["crossover_load"]), (4, None))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "run"
            write_run(root, rates, experiment=tuned, launch={"neo": 0.8, "neo_tuned": 0.8})
            with self.assertRaisesRegex(ValueError, "CPU launch time"):
                load_serving(root)

    def test_no_win_without_saturation(self):
        table = [{"system": s, "memory": "b0064", "load": 4, "language": "eng_Latn", "generated_tokens_per_s": r,
                  "rate_min": r, "rate_max": r, "round_spread_pct": 1.0, "swap_out_requests": 0}
                 for s, r in (("neo", 300.0), ("neo_base", 200.0))]
        self.assertIsNone(crossovers(table, [4])[0]["crossover_load"])

    def test_kv_mismatch_and_missing_round_are_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "run"
            write_run(root, self.RATES, reported=50)
            with self.assertRaisesRegex(ValueError, "KV capacity mismatch"):
                load_serving(root)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "run"
            write_run(root, self.RATES)
            next((root / "cells").glob("round-2-*.json")).unlink()
            with self.assertRaisesRegex(ValueError, "Incomplete run"):
                load_serving(root)


class TreeTests(unittest.TestCase):
    def test_process_tree_includes_children(self):
        child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(5)"])
        try:
            time.sleep(0.2)
            reading = read_process_tree(os.getpid())
            self.assertGreaterEqual(reading["processes"], 2)
            self.assertGreater(reading["rss_mib"], 0)
        finally:
            child.kill()
            child.wait()


if __name__ == "__main__":
    unittest.main()
