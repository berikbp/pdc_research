import json
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

from pdc_research.checkpoints import Checkpoints
from pdc_research.controlled import drive, schedule


class ControlledTests(unittest.TestCase):
    def test_separate_rounds_partial_resume_and_placement(self):
        configs = [{'id': 'gpu', 'threads': 1, 'gpu_layers': 99,
                    'tensor_overrides': ['token_embd.weight=CUDA0']},
                   {'id': 'mixed', 'threads': 4, 'gpu_layers': 24}]
        experiment = {'seed': 42, 'repetitions': 3, 'warmups': 1, 'output_tokens': 64}
        machine = {'request_timeout_seconds': 1}
        cases = [{'group_id': 'a', 'language': lang, 'mode': 'matched-64',
                  'tokens': list(range(64)), 'text': lang} for lang in ['en', 'kk']]
        blocks = schedule(configs, 3, 42)
        self.assertEqual([b['round'] for b in blocks], [1, 1, 2, 2, 3, 3])
        self.assertEqual(blocks, schedule(configs, 3, 42))
        loads, requests = [], []

        @contextmanager
        def fake_server(runtime, threads, layers, log):
            loads.append((runtime['tensor_overrides'], threads, layers))
            yield 'fake'

        def fake_completion(base, tokens, experiment, timeout):
            requests.append(tokens)
            self.assertEqual(len(tokens), 64)
            if len(requests) == 3:  # Interrupt after the first saved measured trial.
                raise KeyboardInterrupt
            return {'timings': {'prompt_n': 64, 'predicted_n': 64, 'cache_n': 0}}, 1

        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            def make_checkpoints():
                result = {}
                for cfg in configs:
                    sub = root / cfg['id']; sub.mkdir(exist_ok=True)
                    result[cfg['id']] = Checkpoints(sub, cases,
                        {'threads': [cfg['threads']], 'gpu_layers': [cfg['gpu_layers']]}, experiment)
                return result
            cp = make_checkpoints()
            with patch('pdc_research.controlled.server', fake_server), patch(
                    'pdc_research.controlled.completion', fake_completion):
                drive(root, machine, experiment, cases, configs, cp, blocks)
                self.assertEqual(json.loads((root/'status.json').read_text())['phase'], 'paused')
                self.assertEqual(sum(len(c.successes) for c in cp.values()), 1)
                cp = make_checkpoints()
                drive(root, machine, experiment, cases, configs, cp, blocks)
                status = json.loads((root/'status.json').read_text())
                self.assertEqual(status['successful'], 12)
                self.assertTrue(status['finished'])
                self.assertEqual(len(loads), 7)  # Interrupted block reloaded, six completed blocks.
                self.assertEqual(sum(bool(c[0]) for c in loads), 3)
                drive(root, machine, experiment, cases, configs, make_checkpoints(), blocks)
                self.assertEqual(len(loads), 7)
