import json
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

from pdc_research.common import load_rows
from pdc_research.backend import completion
from pdc_research.runner import run
from pdc_research.workloads import make_cases
from pdc_research.workloads import prepare


class WorkloadTests(unittest.TestCase):
    def test_sampling_is_aligned_deterministic_and_without_overlap(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "dev").mkdir()
            for lang in ("en", "kk"):
                (root / "dev" / f"{lang}.dev").write_text(
                    "\n".join(f"{lang}-{i}" for i in range(20)), encoding="utf-8")
            config = {"languages": ["en", "kk"], "split": "dev", "groups": 4,
                      "sentences_per_group": 2, "seed": 42}
            prepare(config, root, root / "one.jsonl")
            prepare(config, root, root / "two.jsonl")
            self.assertEqual((root / "one.jsonl").read_bytes(), (root / "two.jsonl").read_bytes())
            rows = load_rows(root / "one.jsonl")
            seen = []
            for index in range(0, len(rows), 2):
                a, b = rows[index:index + 2]
                self.assertEqual(a["sentence_ids"], b["sentence_ids"])
                seen.extend(a["sentence_ids"])
                self.assertEqual(a["text"], "\n".join(f"en-{i - 1}" for i in a["sentence_ids"]))
            self.assertEqual(len(seen), len(set(seen)))
            with self.assertRaises(FileExistsError):
                prepare(config, root, root / "one.jsonl")

    def test_rejects_misaligned_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "dev").mkdir()
            (root / "dev/en.dev").write_text("a\nb\n")
            (root / "dev/kk.dev").write_text("a\n")
            with self.assertRaisesRegex(ValueError, "equal line counts"):
                prepare({"languages": ["en", "kk"], "split": "dev"}, root, root / "out")

    def test_matched_groups_require_every_language_and_never_pad(self):
        rows = [{"group_id": "a", "language": "en", "text": "1234"},
                {"group_id": "a", "language": "kk", "text": "12"},
                {"group_id": "b", "language": "en", "text": "12345"},
                {"group_id": "b", "language": "kk", "text": "1234"}]
        cases = make_cases(rows, {"languages": ["en", "kk"], "matched_tokens": [3]},
                           lambda text: list(range(len(text))))
        matched = [case for case in cases if case["mode"] == "matched-3"]
        self.assertEqual(len(matched), 2)
        self.assertTrue(all(case["group_id"] == "b" and len(case["tokens"]) == 3 for case in matched))


class MeasurementTests(unittest.TestCase):
    def test_runner_records_trials_and_preserves_existing_runs(self):
        @contextmanager
        def fake_server(*args):
            yield "http://fake"

        def fake_request(base, endpoint, payload, timeout):
            if endpoint == "/tokenize":
                return {"tokens": [1, 2, 3]}
            return {"timings": {"prompt_n": len(payload["prompt"]),
                                "predicted_n": payload["n_predict"], "cache_n": 0}}

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "model.gguf").write_bytes(b"fake model")
            (root / "llama-server").write_bytes(b"fake binary")
            manifest = root / "work.jsonl"
            manifest.write_text(json.dumps({"group_id": "a", "language": "en", "text": "abc"}) + "\n")
            machine = {"model": str(root / "model.gguf"), "llama_bin": str(root), "threads": [1],
                       "gpu_layers": [0], "context_size": 20, "request_timeout_seconds": 1}
            experiment = {"languages": ["en"], "repetitions": 2, "warmups": 1,
                          "output_tokens": 2, "seed": 42, "matched_tokens": [2, 8]}
            with patch("pdc_research.runner.server", fake_server), \
                 patch("pdc_research.runner.request", fake_request), \
                 patch("pdc_research.backend.request", fake_request), \
                 patch("pdc_research.runner.capture", return_value={}):
                run(machine, experiment, manifest, root / "result")
                trials = load_rows(root / "result/trials.jsonl")
                self.assertEqual(len(trials), 4)  # Two modes x two repetitions; warm-up excluded.
                self.assertTrue(all(t["status"] == "ok" for t in trials))
                self.assertEqual(json.loads((root / "result/coverage.json").read_text())["matched-8"], 0)
                with self.assertRaises(FileExistsError):
                    run(machine, experiment, manifest, root / "result")

                with patch("pdc_research.runner.completion", side_effect=[({}, 0), ValueError("bad timing")]):
                    with self.assertRaisesRegex(RuntimeError, "1 configurations failed"):
                        run(machine, experiment, manifest, root / "failed")
                failures = load_rows(root / "failed/trials.jsonl")
                trial = next(row for row in failures if row.get("scope") == "trial")
                self.assertEqual(trial["group_id"], "a")
                self.assertEqual(trial["language"], "en")
                self.assertEqual(trial["status"], "error")
                status = json.loads((root / "failed/status.json").read_text())
                self.assertEqual(status["failed_configurations"], 1)

    def test_request_disables_cache_and_checks_fixed_lengths(self):
        response = {"timings": {"prompt_n": 3, "predicted_n": 2, "cache_n": 0}}
        with patch("pdc_research.backend.request", return_value=response) as send:
            completion("http://localhost", [1, 2, 3], {"output_tokens": 2, "seed": 42}, 10)
            payload = send.call_args.args[2]
            self.assertFalse(payload["cache_prompt"])
            self.assertTrue(payload["ignore_eos"])
            self.assertEqual(payload["prompt"], [1, 2, 3])
        for timings in ({"prompt_n": 1, "predicted_n": 2},
                        {"prompt_n": 3, "predicted_n": 1},
                        {"prompt_n": 3, "predicted_n": 2, "cache_n": 1}):
            with patch("pdc_research.backend.request", return_value={"timings": timings}):
                with self.assertRaises(ValueError):
                    completion("http://localhost", [1, 2, 3], {"output_tokens": 2, "seed": 42}, 10)


if __name__ == "__main__":
    unittest.main()
