import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from pdc_research.backend import read_placement, server_command
from pdc_research.config import validate_experiment, validate_machine

MACHINE = {"machine_id": "m", "model": "model.gguf", "llama_bin": "bin", "threads": [8],
           "gpu_layers": [0], "context_size": 2048, "batch_size": 512, "ubatch_size": 128,
           "port": 18080, "startup_timeout_seconds": 1, "request_timeout_seconds": 1}
EXPERIMENT = {"experiment_id": "e", "split": "dev", "languages": ["eng_Latn"], "groups": 1,
              "sentences_per_group": 4, "seed": 1, "repetitions": 1, "warmups": 1,
              "output_tokens": 8, "matched_tokens": [], "synthetic_prompt_tokens": [64],
              "input_mode": "aligned"}


class ServerArgumentTests(unittest.TestCase):
    def test_extra_arguments_follow_fixed_options_and_trace_logging_is_on(self):
        machine = {**MACHINE, "server_args": ["--no-op-offload"]}
        validate_machine(machine)
        with patch("pdc_research.backend.binary", return_value="/fake/llama-server"):
            command = server_command(machine, 8, 0)
        self.assertEqual(command[-1], "--no-op-offload")
        self.assertEqual(command[command.index("-lv") + 1], "4")

    def test_invalid_arguments_are_rejected(self):
        for value in ("--device none", [""], [1], ["-ngl", "5"], ["--port", "1"], ["-t", "2"]):
            with self.assertRaises(ValueError, msg=value):
                validate_machine({**MACHINE, "server_args": value})
        configuration = {"id": "a", "threads": 8, "gpu_layers": 0, "server_args": ["-ngl", "3"]}
        with self.assertRaises(ValueError):
            validate_experiment({**EXPERIMENT, "configurations": [configuration]})
        validate_experiment({**EXPERIMENT, "configurations": [
            {**configuration, "server_args": ["--device", "none"]}]})


class PlacementTests(unittest.TestCase):
    def read(self, text: str) -> dict:
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "server.log"
            log.write_text(text)
            return read_placement(log)

    def test_partial_offload_ignores_unreliable_repeating_line(self):
        self.assertEqual(self.read(
            "0.00.1 I load_tensors: offloading output layer to GPU\n"
            "0.00.1 I load_tensors: offloading 27 repeating layers to GPU\n"
            "0.00.1 I load_tensors: offloaded 29/29 layers to GPU\n"),
            {"offloaded_layers": 29, "total_layers": 29, "blocks_on_gpu": 28, "blocks_on_cpu": 0,
             "output_layer_on_gpu": True})
        self.assertEqual(self.read(
            "0.00.1 I load_tensors: offloading output layer to GPU\n"
            "0.00.1 I load_tensors: offloaded 24/29 layers to GPU\n")["blocks_on_cpu"], 5)

    def test_no_offload(self):
        self.assertEqual(self.read("0.00.1 I load_tensors: offloading 0 repeating layers to GPU\n"
                                   "0.00.1 I load_tensors: offloaded 0/29 layers to GPU\n"),
                         {"offloaded_layers": 0, "total_layers": 29, "blocks_on_gpu": 0,
                          "blocks_on_cpu": 28, "output_layer_on_gpu": False})

    def test_contradictory_lines_are_an_error(self):
        self.assertIn("error", self.read("0.00.1 I load_tensors: offloaded 3/29 layers to GPU\n"))

    def test_missing_report_is_an_error(self):
        self.assertIn("error", self.read("nothing useful\n"))
        self.assertIn("error", read_placement("/nonexistent/server.log"))


if __name__ == "__main__":
    unittest.main()
