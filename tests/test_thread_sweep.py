import tempfile
import unittest
from pathlib import Path
from contextlib import contextmanager
from unittest.mock import patch

from pdc_research.checkpoints import Checkpoints
from pdc_research.runner import _drive


class ThreadSweepTests(unittest.TestCase):
    def test_one_load_per_configuration_and_resume_skips_successes(self):
        loads = []
        calls = []

        @contextmanager
        def fake_server(machine, threads, layers, log):
            loads.append((threads, layers))
            yield 'fake'

        def fake_completion(base, tokens, experiment, timeout):
            calls.append(tokens)
            return {'timings': {}}, 1

        with tempfile.TemporaryDirectory() as directory:
            cases = [{'group_id': group, 'language': 'en', 'mode': 'aligned',
                      'tokens': [1, 2], 'text': group} for group in ('a', 'b')]
            machine = {'threads': [1, 2], 'gpu_layers': [99], 'request_timeout_seconds': 1}
            experiment = {'seed': 42, 'repetitions': 3, 'warmups': 1,
                          'execution_order': 'configuration-first'}
            checkpoints = Checkpoints(Path(directory), cases, machine, experiment)
            with patch('pdc_research.runner.server', fake_server), patch(
                    'pdc_research.runner.completion', fake_completion):
                _drive(checkpoints, machine, experiment, None)
                self.assertEqual(len(loads), 2)
                self.assertEqual(len(calls), 14)  # 12 measured requests + two warmups.
                self.assertEqual(len(checkpoints.successes), 12)
                resumed = Checkpoints(Path(directory), cases, machine, experiment)
                _drive(resumed, machine, experiment, None)
                self.assertEqual(len(loads), 2)
