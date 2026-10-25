"""Equal-length inference in separately shuffled rounds with durable checkpoints."""

import argparse
import random
import time
import uuid
from pathlib import Path

from .backend import binary, completion, server
from .checkpoints import Checkpoints, atomic_json, run_lock
from .common import load_rows, sha256
from .config import load_configs, validate_machine
from .session import build_fingerprint, initialize_run, prepare_cases, check_stop, StopRequested


def schedule(configs, rounds, seed):
    blocks = []
    for repetition in range(rounds):
        order = list(configs)
        random.Random(seed + repetition).shuffle(order)
        blocks.extend({'round': repetition + 1, 'configuration': c['id']} for c in order)
    return blocks


def drive(root, machine, experiment, cases, configs, checkpoints, blocks):
    case_map = {(c['group_id'], c['language'], c['mode']): c for c in cases}
    current = {}

    def progress(phase, message=None):
        counts = {key: len(cp.successes) for key, cp in checkpoints.items()}
        expected = sum(sum(map(len, cp.expected.values())) for cp in checkpoints.values())
        value = {'phase': phase, 'finished': phase == 'complete',
                 'successful': sum(counts.values()), 'expected': expected,
                 'by_configuration': counts, **current}
        if message:
            value['message'] = message
        atomic_json(root / 'status.json', value)
        return value

    try:
        progress('running')
        for block in blocks:
            cid, repetition = block['configuration'], block['round'] - 1
            cfg = next(c for c in configs if c['id'] == cid)
            cp = checkpoints[cid]
            pending = [(ordinal, task) for ordinal in cp.expected for task in cp.pending(ordinal)
                       if task['repetition'] == repetition]
            if not pending:
                continue
            current.update(round=repetition + 1, configuration=cid)
            progress('running')
            check_stop(root)
            runtime = {**machine, 'tensor_overrides': cfg.get('tensor_overrides', [])}
            threads, layers = cfg['threads'], cfg['gpu_layers']
            random.Random(experiment['seed'] + repetition * 1000 + configs.index(cfg)).shuffle(pending)
            log = cp.root / 'logs' / f'round-{repetition + 1}-{uuid.uuid4().hex}.log'
            log.parent.mkdir(parents=True, exist_ok=True)
            started = time.perf_counter()
            with server(runtime, threads, layers, log) as base:
                startup_ms = (time.perf_counter() - started) * 1000
                first = pending[0][1]
                warmup = case_map[first['group_id'], first['language'], first['mode']]
                for _ in range(experiment['warmups']):
                    check_stop(root)
                    completion(base, warmup['tokens'], experiment, machine['request_timeout_seconds'])
                for ordinal, identity in pending:
                    check_stop(root)
                    case = case_map[identity['group_id'], identity['language'], identity['mode']]
                    record = {**identity, 'prompt': ordinal, 'configuration': cid,
                              'round': repetition + 1, 'tensor_overrides': runtime['tensor_overrides'],
                              'server_startup_ms': startup_ms,
                              'server_log': str(log.relative_to(cp.root))}
                    try:
                        response, elapsed = completion(base, case['tokens'], experiment,
                                                       machine['request_timeout_seconds'])
                    except (OSError, ValueError, KeyError, RuntimeError) as exc:
                        cp.save(ordinal, {**record, 'status': 'error', 'error': str(exc)})
                        raise
                    cp.save(ordinal, {**record, 'status': 'ok', 'response': response,
                                      'client_elapsed_ms': elapsed})
                    cp.progress('running', ordinal)
                    status = progress('running')
                    print(f"Round {repetition + 1}, {cid}: {status['successful']}/{status['expected']}", flush=True)
            cp.progress('complete' if not any(cp.pending(n) for n in cp.expected) else 'running')
        if any(cp.pending(n) for cp in checkpoints.values() for n in cp.expected):
            raise RuntimeError('Schedule ended with missing requests')
        progress('complete')
    except (StopRequested, KeyboardInterrupt):
        progress('paused')
        print('Results saved; resume with --resume.', flush=True)
    except (OSError, ValueError, KeyError, RuntimeError) as exc:
        progress('failed', str(exc))
        raise


def run_controlled(machine, experiment, manifest, output, *, resume=False):
    if not experiment.get('configurations'):
        raise ValueError('run-controlled requires an explicit configuration matrix')
    configs = experiment['configurations']
    ids = [c['id'] for c in configs]
    if not ids or len(set(ids)) != len(ids) or any(not s.replace('-', '').isalnum() for s in ids):
        raise ValueError('Configuration IDs must be unique safe names')
    for cfg in configs:
        validate_machine({**machine, 'threads': [cfg['threads']], 'gpu_layers': [cfg['gpu_layers']],
                          'tensor_overrides': cfg.get('tensor_overrides', [])})
    if experiment['matched_tokens'] != [64] or experiment['output_tokens'] != 64:
        raise ValueError('This protocol requires 64 input and 64 output tokens')
    root = Path(output)
    if root.exists() and not resume:
        raise FileExistsError('Output exists; pass --resume')
    if resume and not root.is_dir():
        raise FileNotFoundError('No experiment to resume')
    executable = binary(machine, 'llama-server')
    rows = load_rows(manifest)
    identity = build_fingerprint(machine, experiment, manifest, executable)
    identity['execution_order'] = 'controlled-rounds-v1'
    identity['implementation_sha256'] = {
        name: sha256(Path(__file__).with_name(name))
        for name in ['controlled.py', 'session.py', 'backend.py', 'checkpoints.py', 'workloads.py', 'config.py']
    }
    root.mkdir(parents=True, exist_ok=resume)
    with run_lock(root):
        initialize_run(root, identity, rows, executable, resume)
        atomic_json(root / 'status.json', {'phase': 'preparing', 'finished': False})
        all_cases = prepare_cases(root, machine, experiment, rows)
        cases = [c for c in all_cases if c['mode'] == 'matched-64']
        groups = {c['group_id'] for c in cases}
        if len(cases) != 76 or len(groups) != 19 or any(len(c['tokens']) != 64 for c in cases):
            raise ValueError('Expected exactly 19 eligible groups / 76 equal-length inputs')
        atomic_json(root / 'controlled-cases.json', cases)
        blocks = schedule(configs, experiment['repetitions'], experiment['seed'])
        atomic_json(root / 'schedule.json', blocks)
        checkpoints = {}
        for cfg in configs:
            subroot = root / 'configurations' / cfg['id']
            subroot.mkdir(parents=True, exist_ok=True)
            runtime = {**machine, 'threads': [cfg['threads']], 'gpu_layers': [cfg['gpu_layers']],
                       'tensor_overrides': cfg.get('tensor_overrides', [])}
            checkpoints[cfg['id']] = Checkpoints(subroot, cases, runtime, experiment)
        print(f"Controlled inputs: {len(cases)}; configurations: {len(configs)}; rounds: {experiment['repetitions']}", flush=True)
        drive(root, machine, experiment, cases, configs, checkpoints, blocks)


def read_controlled_status(root):
    """Recount committed results without modifying files or loading the model."""
    from .common import read_json
    root = Path(root)
    metadata = read_json(root / 'metadata.json')
    previous = read_json(root / 'status.json') if (root / 'status.json').exists() else {}
    if not metadata.get('cases_sha256'):
        return previous
    if sha256(root / 'cases.json') != metadata['cases_sha256']:
        raise ValueError('Saved cases checksum mismatch')
    cases = [c for c in read_json(root / 'cases.json') if c['mode'] == 'matched-64']
    fingerprint = metadata['fingerprint']
    experiment, machine = fingerprint['experiment'], fingerprint['machine']
    counts, expected = {}, 0
    for cfg in experiment['configurations']:
        runtime = {**machine, 'threads': [cfg['threads']], 'gpu_layers': [cfg['gpu_layers']]}
        cp = Checkpoints(root / 'configurations' / cfg['id'], cases, runtime, experiment)
        for record in cp.successes.values():
            if record.get('configuration') != cfg['id'] or record.get('tensor_overrides') != cfg.get('tensor_overrides', []):
                raise ValueError('Result configuration placement mismatch')
        counts[cfg['id']] = len(cp.successes)
        expected += sum(map(len, cp.expected.values()))
    successful = sum(counts.values())
    phase = previous.get('phase', 'unknown')
    if phase == 'complete' and successful != expected:
        phase = 'incomplete'
    return {**previous, 'phase': phase, 'finished': phase == 'complete',
            'successful': successful, 'expected': expected, 'by_configuration': counts}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--machine', required=True)
    parser.add_argument('--experiment', required=True)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    machine, experiment = load_configs(args.machine, args.experiment)
    run_controlled(machine, experiment, args.manifest, args.output, resume=args.resume)


if __name__ == '__main__':
    main()
