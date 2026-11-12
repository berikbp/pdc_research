"""A model batch must not overlap workloads or advance past a deliberate pause."""
import importlib.util
import json
import subprocess
import tempfile
import unittest
from pathlib import Path

spec=importlib.util.spec_from_file_location('run_model_batch',Path('scripts/run_model_batch.py'))
batch=importlib.util.module_from_spec(spec)
spec.loader.exec_module(batch)


class ModelBatchTests(unittest.TestCase):
    def fixture(self,root):
        models=[{'id':name,'assets':f'{name}.json','machine':f'{name}.machine.json',
                 'output':str(root/name),'pipeline':str(root/f'{name}-pipeline')} for name in ['a','b']]
        path=root/'config.json'
        path.write_text(json.dumps({'batch_id':'test','experiment':'experiment.json',
                                     'manifest':'inputs.jsonl','models':models}))
        return path,models

    def test_models_are_downloaded_then_benchmarked_in_order(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);config,models=self.fixture(root);calls=[]
            def execute(cmd,check):
                self.assertTrue(check);calls.append(cmd[1])
                if cmd[1].endswith('run_model_extension.py'):
                    p=Path(cmd[cmd.index('--pipeline')+1]);p.mkdir()
                    (p/'status.json').write_text(json.dumps({'phase':'complete','report':'report.md'}))
            batch.run_batch(config,root/'batch',execute=execute)
            self.assertEqual(calls,['scripts/download_assets.py','scripts/run_model_extension.py']*2)
            self.assertEqual(json.loads((root/'batch/status.json').read_text())['phase'],'complete')

    def test_pause_stops_before_next_model(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);config,models=self.fixture(root);calls=[]
            def execute(cmd,check):
                calls.append(cmd[1])
                if cmd[1].endswith('run_model_extension.py'):
                    p=Path(cmd[cmd.index('--pipeline')+1]);p.mkdir()
                    (p/'status.json').write_text(json.dumps({'phase':'paused'}))
            batch.run_batch(config,root/'batch',execute=execute)
            self.assertEqual(len(calls),2)
            status=json.loads((root/'batch/status.json').read_text())
            self.assertEqual(status['phase'],'paused')
            self.assertEqual(status['models'][1]['phase'],'pending')

    def test_failure_is_retained_and_next_model_is_attempted(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);config,models=self.fixture(root);calls=[]
            def execute(cmd,check):
                calls.append(cmd)
                if cmd[1].endswith('download_assets.py') and cmd[cmd.index('--config')+1]=='a.json':
                    raise subprocess.CalledProcessError(1,cmd)
                if cmd[1].endswith('run_model_extension.py'):
                    p=Path(cmd[cmd.index('--pipeline')+1]);p.mkdir()
                    (p/'status.json').write_text(json.dumps({'phase':'complete','report':'report.md'}))
            with self.assertRaises(RuntimeError):
                batch.run_batch(config,root/'batch',execute=execute)
            self.assertEqual(len(calls),3)
            status=json.loads((root/'batch/status.json').read_text())
            self.assertEqual(status['phase'],'complete-with-failures')
            self.assertEqual([r['phase'] for r in status['models']],['failed','complete'])
