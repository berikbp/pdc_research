"""Model extensions must preserve natural inputs and verify single-file assets."""
import importlib.util
import json
import tempfile
import unittest
from contextlib import chdir
from pathlib import Path
from unittest.mock import patch

from pdc_research.common import sha256
from pdc_research.config import load_configs, experiment_plan
from pdc_research.controlled import select_cases


def script(name):
    spec = importlib.util.spec_from_file_location(name, Path('scripts')/f'{name}.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ModelExtensionTests(unittest.TestCase):
    def test_aligned_selection_keeps_full_tokens_and_historical_default_is_matched(self):
        cases = [{'group_id': group, 'language': lang, 'mode': mode, 'tokens': [1]*length}
                 for group in ['a','b'] for lang in ['en','kk']
                 for mode,length in [('aligned',100 if lang=='kk' else 75),('matched-64',64)]]
        experiment = {'groups':2,'languages':['en','kk'],'input_mode':'aligned','expected_groups':2}
        selected = select_cases(cases, experiment)
        self.assertEqual(len(selected),4)
        self.assertEqual({len(c['tokens']) for c in selected},{75,100})
        matched = select_cases(cases,{**experiment,'input_mode':'matched-64'})
        self.assertEqual({len(c['tokens']) for c in matched},{64})
        with self.assertRaisesRegex(ValueError,'Expected'):
            select_cases(cases,{**experiment,'expected_groups':3})

    def test_complete_model_experiment_budget_and_explicit_placements(self):
        machine,experiment=load_configs('configs/machines/berik-qwen3b.json',
                                         'configs/experiments/model-thread-rounds.json')
        self.assertEqual(experiment_plan(machine,experiment)['expected_measured_requests'],2160)
        cpu=[c for c in experiment['configurations'] if c['id'].startswith('cpu-emb')]
        self.assertEqual([c['threads'] for c in cpu],[1,2,4,6,8,10,12,16])
        self.assertTrue(all(c['tensor_overrides']==['token_embd.weight=CPU'] for c in cpu))
        self.assertEqual(experiment['configurations'][-1]['tensor_overrides'],['token_embd.weight=CUDA0'])

    def test_single_file_model_verifies_without_merge(self):
        assets=script('download_assets')
        with tempfile.TemporaryDirectory() as directory, chdir(directory):
            path=Path('models/small.gguf');path.parent.mkdir();path.write_bytes(b'fake-GGUF')
            model={'repository':'example/model','revision':'a'*40,'merged_filename':path.name,
                   'files':[{'filename':path.name,'size_bytes':path.stat().st_size,'sha256':sha256(path)}]}
            with patch.object(assets.subprocess,'run') as process:
                assets.prepare_model({'model':model})
                assets.prepare_model({'model':model})
                process.assert_not_called()
            record=json.loads(path.with_suffix('.metadata.json').read_text())
            self.assertEqual(record['merged_sha256'],sha256(path))
            self.assertEqual(record['format'],'single-file-gguf')
            path.write_bytes(b'corrupted')
            with self.assertRaises(ValueError):assets.prepare_model({'model':model})

    def test_placement_parser_distinguishes_compute_from_reservation_nodes(self):
        verification=script('verify_model_placement')
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'graph.log'
            path.write_text('D node # 0 ( GET_ROWS): embd ( 14K) [ CPU ] use=2,c=1:\n'
                            'D node # 1 ( GET_ROWS): unused ( 14K) [ CPU ] use=0,c=0:\n'
                            'D node # 2 ( MUL_MAT): layer ( 14K) [CUDA0 ] use=1,c=1:\n')
            self.assertEqual(verification.placements(path),[('GET_ROWS','CPU'),('MUL_MAT','CUDA0')])

    def test_thread_analysis_accepts_aligned_balanced_rounds(self):
        from pdc_research.checkpoints import Checkpoints,atomic_json
        from pdc_research.thread_analysis import analyze_thread_sweep
        from pdc_research.analysis import intervals
        cases=[{'group_id':g,'language':l,'mode':'aligned','text':'example',
                'tokens':[1]*(75 if l=='en' else 100)} for g in ['a','b'] for l in ['en','kk']]
        configs=[{'id':'cpu-emb-t1','threads':1,'gpu_layers':99,'tensor_overrides':['token_embd.weight=CPU']},
                 {'id':'gpu-model-only','threads':1,'gpu_layers':99,'tensor_overrides':['token_embd.weight=CUDA0']}]
        experiment={'groups':2,'expected_groups':2,'input_mode':'aligned','languages':['en','kk'],
                    'repetitions':3,'output_tokens':64,'configurations':configs}
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            atomic_json(root/'cases.json',cases)
            atomic_json(root/'metadata.json',{'cases_sha256':sha256(root/'cases.json'),
                'fingerprint':{'schema_version':2,'execution_order':'controlled-rounds-v1',
                               'model_sha256':'fake','machine':{'model':'fake'},'experiment':experiment}})
            atomic_json(root/'status.json',{'phase':'complete','finished':True})
            for cfg in configs:
                cp=Checkpoints(root/'configurations'/cfg['id'],cases,
                               {'threads':[1],'gpu_layers':[99]},experiment)
                for n in cp.expected:
                    for task in cp.pending(n):
                        cp.save(n,{**task,'configuration':cfg['id'],'round':task['repetition']+1,
                                   'tensor_overrides':cfg['tensor_overrides'],'status':'ok',
                                   'response':{'timings':{'prompt_n':task['input_tokens'],'predicted_n':64,
                                                         'cache_n':0,'prompt_ms':100,'predicted_ms':1000}}})
            with patch('pdc_research.thread_analysis.intervals',side_effect=lambda v:intervals(v,samples=100)):
                analyze_thread_sweep(root,root/'analysis')
            self.assertIn('Validated 24 unique trials',(root/'analysis/report.md').read_text())
