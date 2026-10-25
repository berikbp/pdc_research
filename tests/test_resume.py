"""Resume must skip successful work, preserve errors, and respect group boundaries."""

import json
import tempfile
import unittest
from contextlib import contextmanager, ExitStack
from pathlib import Path
from unittest.mock import patch

from pdc_research.checkpoints import atomic_json, read_status, run_lock
from pdc_research.common import read_json
from pdc_research.runner import run


class ResumeTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        (self.root / "model.gguf").write_bytes(b"fake model")
        (self.root / "llama-server").write_bytes(b"fake backend")
        self.machine = {
            "model": str(self.root / "model.gguf"), "llama_bin": str(self.root),
            "threads": [1], "gpu_layers": [0, 1], "context_size": 20,
            "request_timeout_seconds": 1,
        }
        self.experiment = {
            "languages": ["en", "kk"], "repetitions": 1, "warmups": 1,
            "output_tokens": 2, "seed": 42, "matched_tokens": [],
        }
        self.manifest = self.root / "inputs.jsonl"
        rows = [{"group_id": group, "language": language, "text": f"{group}-{language}"}
                for group in ("a", "b") for language in ("en", "kk")]
        self.manifest.write_text("".join(json.dumps(row) + "\n" for row in rows))
        self.output = self.root / "run"

    @contextmanager
    def backend(self, side_effect=None):
        @contextmanager
        def fake_server(*args):
            yield "http://fake"

        def fake_request(*args):
            return {"tokens": [1, 2, 3]}

        def fake_completion(*args):
            return {"content": "result", "timings": {"prompt_n": 3, "predicted_n": 2}}, 1.0

        with ExitStack() as stack:
            server_mock = stack.enter_context(patch("pdc_research.runner.server", side_effect=fake_server))
            stack.enter_context(patch("pdc_research.session.server", side_effect=fake_server))
            stack.enter_context(patch("pdc_research.session.request", side_effect=fake_request))
            stack.enter_context(patch("pdc_research.session.capture", return_value={}))
            completion_mock = stack.enter_context(patch(
                "pdc_research.runner.completion", side_effect=side_effect or fake_completion
            ))
            yield server_mock, completion_mock

    def execute(self, **kwargs):
        run(self.machine, self.experiment, self.manifest, self.output, **kwargs)

    def results(self):
        return sorted(self.output.glob("prompts/*/results/*.json"))

    def test_stop_at_prompt_boundary_then_resume_next_prompt(self):
        with self.backend():
            self.execute(stop_after_prompts=1)
        before = {path: path.read_bytes() for path in self.results()}
        status = read_json(self.output / "status.json")
        self.assertEqual(status["phase"], "paused")
        self.assertEqual(status["successful"], 4)
        self.assertEqual(status["next_prompt"], 2)
        with self.backend() as (_, complete):
            self.execute(resume=True)
            self.assertEqual(complete.call_count, 6)  # Four new trials + two warm-ups.
        self.assertEqual(len(self.results()), 8)
        for path, content in before.items():
            self.assertEqual(path.read_bytes(), content)
        self.assertEqual(read_json(self.output / "status.json")["phase"], "complete")

    def test_interrupt_mid_prompt_preserves_success_and_retries_missing_only(self):
        counter = 0

        def interrupt(*args):
            nonlocal counter
            counter += 1
            if counter == 3:  # Warm-up, one saved trial, then interrupted request.
                raise KeyboardInterrupt
            return {"content": "result", "timings": {}}, 1.0

        with self.backend(interrupt):
            with self.assertRaises(KeyboardInterrupt):
                self.execute()
        self.assertEqual(len(self.results()), 1)
        self.assertEqual(read_json(self.output / "status.json")["next_prompt"], 1)
        preserved = self.results()[0].read_bytes()
        with self.backend() as (_, complete):
            self.execute(resume=True)
            self.assertEqual(complete.call_count, 11)  # Seven trials + four configuration warm-ups.
        self.assertEqual(len(self.results()), 8)
        self.assertIn(preserved, [path.read_bytes() for path in self.results()])

    def test_complete_run_resumes_without_launching_any_server(self):
        with self.backend():
            self.execute()
        with self.backend() as (server_mock, completion_mock):
            self.execute(resume=True)
            server_mock.assert_not_called()
            completion_mock.assert_not_called()

    def test_status_reconstructs_saved_results_without_changing_run(self):
        with self.backend():
            self.execute(stop_after_prompts=1)
        atomic_json(self.output / "status.json", {"phase": "paused", "successful": 0})
        before = (self.output / "status.json").read_bytes()
        status = read_status(self.output)
        self.assertEqual(status["successful"], 4)
        self.assertEqual(status["next_prompt"], 2)
        self.assertEqual((self.output / "status.json").read_bytes(), before)

    def test_failed_prompt_never_advances_to_next_group(self):
        def fail(*args):
            raise ValueError("backend failure")
        with self.backend(fail):
            with self.assertRaisesRegex(RuntimeError, "Prompt 1 incomplete"):
                self.execute()
        self.assertFalse((self.output / "prompts/002").exists())
        status = read_status(self.output)
        self.assertEqual(status["next_prompt"], 1)
        self.assertEqual(status["successful"], 0)

    def test_changed_model_or_settings_refuse_resume(self):
        with self.backend():
            self.execute(stop_after_prompts=1)
        with self.backend():
            self.experiment["output_tokens"] = 3
            with self.assertRaisesRegex(ValueError, "Resume refused"):
                self.execute(resume=True)
            self.experiment["output_tokens"] = 2
            (self.root / "model.gguf").write_bytes(b"changed model")
            with self.assertRaisesRegex(ValueError, "Resume refused"):
                self.execute(resume=True)

    def test_stop_file_finishes_current_trial_and_resume_acknowledges_stop(self):
        counter = 0

        def stop(*args):
            nonlocal counter
            counter += 1
            if counter == 2:
                (self.output / "STOP").touch()
            return {"content": "result", "timings": {}}, 1.0

        with self.backend(stop):
            self.execute()
        self.assertEqual(len(self.results()), 1)
        self.assertEqual(read_json(self.output / "status.json")["phase"], "paused")
        with self.backend():
            self.execute(resume=True)
        self.assertFalse((self.output / "STOP").exists())
        self.assertEqual(len(self.results()), 8)

    def test_atomic_write_failure_preserves_previous_checkpoint(self):
        path = self.root / "checkpoint.json"
        atomic_json(path, {"successful": 1})
        with patch("pdc_research.checkpoints.os.replace", side_effect=OSError("disk error")):
            with self.assertRaises(OSError):
                atomic_json(path, {"successful": 2})
        self.assertEqual(read_json(path), {"successful": 1})

    def test_run_directory_cannot_have_two_writers(self):
        with run_lock(self.root):
            with self.assertRaisesRegex(RuntimeError, "already using"):
                with run_lock(self.root):
                    self.fail("Acquired a second lock")


if __name__ == "__main__":
    unittest.main()
