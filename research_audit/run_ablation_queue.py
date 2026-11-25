"""Run fixed development ablations sequentially on one assigned GPU."""
import argparse
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
    work = root / 'ablation_study'
    if args.wait_pid:
        print(f'Waiting for existing PID {args.wait_pid}', flush=True)
        while Path(f'/proc/{args.wait_pid}').exists():
            time.sleep(10)
    if args.gpu == 0:
        jobs = [('no_kl', 42), ('single_no_kl', 42), ('no_rank', 42),
                ('no_kl', 456), ('single_no_kl', 456)]
    else:
        jobs = [('no_kl', 123), ('single_no_kl', 123), ('no_rank', 123),
                ('no_rank', 456), ('three_scale', 42)]
    env = {**os.environ, 'CUDA_VISIBLE_DEVICES': str(args.gpu),
           'PYTHONPATH': str(root / 'deps') + os.pathsep + str(work)}
    results = {}
    for arm, seed in jobs:
        name = f'{arm}_seed{seed}'
        scales = [224] if arm == 'single_no_kl' else [224, 384, 512] if arm == 'three_scale' else [224, 384]
        kl = 0.0 if arm in ('no_kl', 'single_no_kl') else 0.10
        rank = 0.0 if arm == 'no_rank' else 0.5
        command = [sys.executable, '-u', '-m', 'ct_musiq.train', '--epochs', '100',
                   '--batch_size', '32', '--lambda_kl', str(kl), '--lambda-rank', str(rank),
                   '--split-manifest', 'research_audit/development_split_seed42.json',
                   '--output-dir', f'results/{name}', '--seed', str(seed), '--num-workers', '4',
                   '--pretrained-file', str(root / 'vit_base_patch32_224.safetensors'),
                   '--scales', *map(str, scales)]
        print(f'Starting {name} on GPU {args.gpu}', flush=True)
        with (root / f'ablation_{name}.log').open('w') as log:
            result = subprocess.run(command, cwd=work, env=env, stdout=log, stderr=subprocess.STDOUT)
        results[name] = {'exit_code': result.returncode, 'gpu': args.gpu,
                         'seed': seed, 'scales': scales, 'lambda_kl': kl, 'lambda_rank': rank}
        status = root / f'ablation_queue_gpu{args.gpu}.json'
        temp = status.with_suffix('.tmp')
        temp.write_text(json.dumps(results, indent=2) + '\n')
        temp.replace(status)
        print(f'Finished {name}: exit {result.returncode}', flush=True)


if __name__ == '__main__':
    main()
