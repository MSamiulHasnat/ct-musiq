"""Run the frozen compact replications, then evaluate every declared screen.

Copy this launcher to the remote experiment root; no network downloads occur.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, choices=[0, 1], required=True)
    parser.add_argument('--wait-pid', type=int)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    work = root / 'development_screening'
    expected = '0dbf09ac9e6fe934c85e0329cbd205eea0323546a380064c78d27b45e686e6c0'
    if hashlib.sha256((root / 'development-screening-source.tar.gz').read_bytes()).hexdigest() != expected:
        raise RuntimeError('Frozen screening source archive does not match protocol')
    if args.wait_pid:
        print(f'Waiting for existing screen PID {args.wait_pid}', flush=True)
        while Path(f'/proc/{args.wait_pid}').exists():
            time.sleep(10)
    env = {**os.environ, 'CUDA_VISIBLE_DEVICES': str(args.gpu),
           'PYTHONPATH': str(root / 'deps') + os.pathsep + str(work)}
    seed = 456 if args.gpu == 0 else 123
    status = {}

    def run(name, command):
        started = time.monotonic()
        print(f'Starting {name} on GPU {args.gpu}', flush=True)
        with (root / f'{name}.log').open('x') as log:
            result = subprocess.run([sys.executable, '-u', '-m', *command], cwd=work,
                                    env=env, stdout=log, stderr=subprocess.STDOUT)
        status[name] = {'exit_code': result.returncode, 'wall_seconds': time.monotonic()-started,
                        'gpu': args.gpu, 'command': command}
        (root / f'screen_followup_gpu{args.gpu}.json').write_text(json.dumps(status, indent=2))
        if result.returncode:
            raise RuntimeError(f'{name} failed with exit {result.returncode}')
        print(f'Completed {name}', flush=True)

    run(f'screen_compact448_seed{seed}', ['research_audit.compact_iqa',
        '--image-dir', 'dataset/image', '--labels', 'dataset/train.json',
        '--split', 'research_audit/development_split_seed42.json',
        '--pretrained-file', '../efficientnet_b0.safetensors',
        '--output', f'results/compact448_seed{seed}', '--resolution', '448',
        '--batch-size', '32', '--epochs', '100', '--seed', str(seed), '--workers', '4'])
    for eval_seed in ([456] if args.gpu == 0 else [42, 123]):
        run(f'eval_compact448_seed{eval_seed}', ['research_audit.compact_iqa',
            '--image-dir', '../LDCTIQAC_test/images', '--labels', '../LDCTIQAC_test/test.json',
            '--checkpoint', f'results/compact448_seed{eval_seed}/best.pth',
            '--output', f'../evaluations/compact448_seed{eval_seed}', '--batch-size', '32', '--workers', '4'])
    if args.gpu == 0:
        run('eval_aligned_seed42', ['research_audit.evaluate_released_test',
            '--data', '../LDCTIQAC_test', '--checkpoint', 'results/aligned_seed42/ct_musiq/ct_musiq_best.pth',
            '--output', '../evaluations/aligned_seed42', '--batch-size', '32'])


if __name__ == '__main__':
    main()
