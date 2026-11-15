import json
import tempfile
import unittest
from pathlib import Path

from pdc_research.common import sha256
from pdc_research.layer_analysis import analyze_layer_sweep, fit, load_trials, solve

TOTAL = 29  # 28 blocks + output layer
BASE, PER_BLOCK, HANDOFF, OUTPUT = 20.0, 2.5, 1.5, 0.7


def write_run(root: Path, layers: list[int], drop_last: bool = False) -> None:
    groups, languages = ["dev-0001", "dev-0002", "dev-0003"], ["eng_Latn", "kaz_Cyrl"]
    lengths = {"eng_Latn": 90, "kaz_Cyrl": 300}
    experiment = {"experiment_id": "synthetic", "input_mode": "aligned", "languages": languages,
                  "repetitions": 2, "output_tokens": 64,
                  "configurations": [{"id": f"g{n:02d}", "threads": 8, "gpu_layers": n} for n in layers]}
    cases = [{"group_id": g, "language": lang, "mode": "aligned", "tokens": list(range(lengths[lang]))}
             for g in groups for lang in languages]
    (root / "cases.json").write_text(json.dumps(cases))
    (root / "metadata.json").write_text(json.dumps(
        {"fingerprint": {"experiment": experiment}, "cases_sha256": sha256(root / "cases.json")}))
    records = []
    for n in layers:
        offloaded = min(n, TOTAL)
        on_cpu = TOTAL - 1 - max(offloaded - 1, 0)
        per_token = BASE + PER_BLOCK * on_cpu + HANDOFF * (on_cpu > 0) + OUTPUT * (offloaded == 0)
        for index, group in enumerate(groups):
            for lang in languages:
                for round_ in (1, 2):
                    noise = 0.01 * (index - 1) * round_
                    records.append({
                        "configuration": f"g{n:02d}", "group_id": group, "language": lang, "round": round_,
                        "status": "ok", "gpu_layers": n, "threads": 8, "server_args": [],
                        "placement": {"offloaded_layers": offloaded, "total_layers": TOTAL,
                                      "blocks_on_gpu": max(offloaded - 1, 0), "blocks_on_cpu": on_cpu,
                                      "output_layer_on_gpu": offloaded >= 1},
                        "telemetry": {"gpu": {"gpu_util_mean_pct": 50.0, "power_mean_w": 40.0, "energy_j": 80.0,
                                              "temp_max_c": 60.0},
                                      "server_gpu_memory": {"vram_mib": 150 * offloaded},
                                      "process": {"cpu_cores_mean": 4.0, "rss_mib": 900.0}},
                        "response": {"timings": {"prompt_n": lengths[lang], "predicted_n": 64, "cache_n": 0,
                                                 "prompt_ms": lengths[lang] * (0.5 + 0.1 * on_cpu),
                                                 "predicted_ms": 64 * (per_token + noise)}}})
    for number, record in enumerate(records[:-1] if drop_last else records):
        path = root / "configurations" / record["configuration"] / "prompts" / "001" / "results" / f"{number}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(record))


class FitTests(unittest.TestCase):
    def test_solve(self):
        self.assertEqual([round(x, 9) for x in solve([[2.0, 1.0], [1.0, 3.0]], [5.0, 10.0])], [1.0, 3.0])
        with self.assertRaises(ValueError):
            solve([[1.0, 1.0], [1.0, 1.0]], [1.0, 2.0])

    def test_terms_without_variation_are_dropped(self):
        result = fit([(k, True, 10 + 2 * k) for k in (1, 2, 3, 4)])
        self.assertEqual(result["fitted_terms"], ["base_ms", "per_cpu_block_ms"])
        self.assertAlmostEqual(result["per_cpu_block_ms"], 2.0)
        self.assertEqual(result["handoff_ms"], 0.0)


class LayerSweepTests(unittest.TestCase):
    def test_recovers_known_cost_model_and_writes_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root, output = Path(directory) / "run", Path(directory) / "out"
            root.mkdir()
            write_run(root, list(range(30)))
            analyze_layer_sweep(root, output)
            model = json.loads((output / "cost-model.json").read_text())["decode_ms_per_token"]
            for term, truth in zip(("base_ms", "per_cpu_block_ms", "handoff_ms", "output_on_cpu_ms"),
                                   (BASE, PER_BLOCK, HANDOFF, OUTPUT)):
                self.assertAlmostEqual(model[term], truth, places=2)
                self.assertLessEqual(model["ci95"][term][0], model[term] + 1e-9)
                self.assertGreaterEqual(model["ci95"][term][1], model[term] - 1e-9)
            self.assertGreater(model["r_squared"], 0.9999)
            layers = (output / "layers.csv").read_text().splitlines()
            self.assertEqual(len(layers), 31)
            report = (output / "report.md").read_text()
            self.assertIn("Each block moved to CPU", report)
            self.assertIn("Total kaz_Cyrl, s", report)

    def test_incomplete_run_is_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_run(root, [0, 29], drop_last=True)
            with self.assertRaisesRegex(ValueError, "Incomplete run"):
                load_trials(root)

    def test_placement_mismatch_is_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_run(root, [0, 29])
            path = next(root.glob("configurations/g29/prompts/001/results/*.json"))
            record = json.loads(path.read_text())
            record["placement"]["offloaded_layers"] = 28
            path.write_text(json.dumps(record))
            with self.assertRaisesRegex(ValueError, "placement mismatch"):
                load_trials(root)


if __name__ == "__main__":
    unittest.main()
