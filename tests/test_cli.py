"""Public entrypoints should route both run formats without loading assets."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from pdc_research.cli import build_parser, dispatch
from pdc_research.checkpoints import Checkpoints, atomic_json, read_status
from pdc_research.common import sha256
from pdc_research.config import load_configs, experiment_plan, validate_experiment


class CliTests(unittest.TestCase):
    def test_controlled_dispatch_and_budget_bound(self):
        args = build_parser().parse_args([
            'run-controlled', '--machine', 'configs/machines/berik-full-gpu.json',
            '--experiment', 'configs/experiments/rq1-controlled.json', '--output', 'runs/test'])
        with patch('pdc_research.cli.run_controlled') as run:
            dispatch(args)
        run.assert_called_once()
        machine, experiment = load_configs(args.machine, args.experiment)
        plan = experiment_plan(machine, experiment)
        self.assertEqual(plan['configurations'], 9)
        self.assertEqual(plan['configuration_loads'], 27)
        self.assertEqual(plan['max_measured_requests'], 2160)  # Eligibility reduces the historical run to 2052.
        for configs in ([], [{'id': '../bad', 'threads': 1, 'gpu_layers': 99}],
                        [{'id': 'x', 'threads': 0, 'gpu_layers': 99}]):
            with self.assertRaises(ValueError):
                validate_experiment({**experiment, 'configurations': configs})

    def test_controlled_status_recounts_without_writing(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cases = [{'group_id': 'a', 'language': 'en', 'text': 'x',
                      'mode': 'matched-64', 'tokens': [1]*64}]
            machine = {'threads': [1], 'gpu_layers': [99]}
            experiment = {'repetitions': 3, 'configurations': [
                {'id': 'gpu', 'threads': 1, 'gpu_layers': 99}]}
            atomic_json(root/'cases.json', cases)
            atomic_json(root/'metadata.json', {'cases_sha256': sha256(root/'cases.json'),
                'fingerprint': {'schema_version': 2, 'execution_order': 'controlled-rounds-v1',
                                'machine': machine, 'experiment': experiment}})
            atomic_json(root/'status.json', {'phase': 'complete', 'successful': 999})
            cp = Checkpoints(root/'configurations/gpu', cases, machine, experiment)
            cp.save(1, {**cp.pending(1)[0], 'configuration': 'gpu', 'tensor_overrides': [], 'status': 'ok'})
            before = {p: p.read_bytes() for p in root.rglob('*') if p.is_file()}
            status = read_status(root)
            self.assertEqual(status['successful'], 1)
            self.assertEqual(status['expected'], 3)
            self.assertEqual(status['phase'], 'incomplete')
            self.assertEqual(before, {p: p.read_bytes() for p in root.rglob('*') if p.is_file()})
