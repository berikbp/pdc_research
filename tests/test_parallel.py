import json
import tempfile
import unittest
from pathlib import Path

from pdc_research.parallel import (analyze_parallel, load_cells, request_order, summarize_block,
                                   validate_parallel)

EXPERIMENT = {"experiment_id": "synthetic", "seed": 1, "output_tokens": 64, "rounds": 2,
              "requests_per_language": 8, "threads": 8,
              "cells": [{"id": "g29-np04", "gpu_layers": 29, "slots": 4, "context_per_slot": 512,
                         "languages": ["eng_Latn", "kaz_Cyrl"]}]}


def block(rate: float) -> dict:
    return {"requests": 8, "wall_s": 512 / rate, "requests_per_s": rate / 64, "generated_tokens_per_s": rate,
            "all_tokens_per_s": rate * 2, "mean_input_tokens": 90.0, "mean_latency_s": 2.0,
            "max_latency_s": 3.0,
            "telemetry": {"gpu": {"gpu_util_mean_pct": 80.0, "power_mean_w": 70.0},
                          "process": {"cpu_cores_mean": 1.0}}}


def write_run(root: Path, offloaded: int = 29) -> None:
    (root / "cells").mkdir(parents=True)
    (root / "metadata.json").write_text(json.dumps({"fingerprint": {"experiment": EXPERIMENT}}))
    for round_, rates in ((1, (100.0, 80.0)), (2, (110.0, 80.0))):
        record = {"status": "ok", "context_size": 2048, "server_vram_after_load": {"vram_mib": 4400},
                  "placement": {"offloaded_layers": offloaded, "total_layers": 29, "blocks_on_cpu": 0},
                  "languages": {"eng_Latn": block(rates[0]), "kaz_Cyrl": block(rates[1])}}
        (root / "cells" / f"round-{round_}-g29-np04.json").write_text(json.dumps(record))


class ParallelTests(unittest.TestCase):
    def test_request_order_covers_passages_before_repeating(self):
        order = request_order(20, 32, seed=7)
        self.assertEqual(len(order), 32)
        self.assertEqual(sorted(order[:20]), list(range(20)))
        self.assertEqual(order, request_order(20, 32, seed=7))
        self.assertNotEqual(order, request_order(20, 32, seed=8))

    def test_block_throughput_uses_wall_time_of_overlapping_requests(self):
        requests = [{"started_s": start, "elapsed_ms": 2000.0, "prompt_n": 100, "predicted_n": 64,
                     "prompt_ms": 50.0, "predicted_ms": 1900.0} for start in (0.0, 0.0, 2.0, 2.0)]
        summary = summarize_block(requests)
        self.assertAlmostEqual(summary["wall_s"], 4.0)
        self.assertAlmostEqual(summary["generated_tokens_per_s"], 64.0)
        self.assertAlmostEqual(summary["requests_per_s"], 1.0)
        self.assertAlmostEqual(summary["mean_latency_s"], 2.0)

    def test_requests_must_fill_every_wave(self):
        validate_parallel(EXPERIMENT)
        with self.assertRaisesRegex(ValueError, "multiple of slots"):
            validate_parallel({**EXPERIMENT, "requests_per_language": 10})

    def test_analysis_averages_rounds_and_reports_spread(self):
        with tempfile.TemporaryDirectory() as directory:
            root, output = Path(directory) / "run", Path(directory) / "out"
            write_run(root)
            analyze_parallel(root, output)
            lines = (output / "cells.csv").read_text().splitlines()
            self.assertEqual(len(lines), 3)
            english = dict(zip(lines[0].split(","), lines[1].split(",")))
            self.assertAlmostEqual(float(english["generated_tokens_per_s"]), 105.0)
            self.assertAlmostEqual(float(english["round_spread_pct"]), 100 * 10 / 105)
            self.assertIn("| g29-np04 |", (output / "report.md").read_text())

    def test_missing_round_and_wrong_placement_are_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "run"
            write_run(root, offloaded=28)
            with self.assertRaisesRegex(ValueError, "Placement mismatch"):
                load_cells(root)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "run"
            write_run(root)
            (root / "cells" / "round-2-g29-np04.json").unlink()
            with self.assertRaisesRegex(ValueError, "Incomplete run"):
                load_cells(root)


if __name__ == "__main__":
    unittest.main()
