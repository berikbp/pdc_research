import json
import unittest
from unittest.mock import patch
from pathlib import Path
from pdc_research.backend import server_command
from pdc_research.config import validate_machine


class GpuBaselineTests(unittest.TestCase):
    def test_embedding_override_is_passed_and_validated(self):
        machine = json.loads(Path('configs/machines/berik-gpu-model-baseline.json').read_text())
        validate_machine(machine)
        with patch("pdc_research.backend.binary", return_value="/fake/llama-server"):
            command = server_command(machine, 1, 99)
        self.assertEqual(command[command.index('-ot') + 1], 'token_embd.weight=CUDA0')
        self.assertEqual(command[command.index('-t') + 1], '1')
        self.assertEqual(command[command.index('-ngl') + 1], '99')
        machine['tensor_overrides'] = ['token_embd.weight=']
        with self.assertRaises(ValueError):
            validate_machine(machine)
