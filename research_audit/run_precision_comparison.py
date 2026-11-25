"""Run only the declared three-seed input-precision control on two GPUs."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    root = Path(__file__).resolve().parent
    work = root/'precision_comparison'
    archive_hash = '3e320878df458159cb496beeedff2f7a88ff6c79c6914b208ad864708c6e5dae'
    if hashlib.sha256((root/'precision-comparison-source.tar.gz').read_bytes()).hexdigest() != archive_hash:
        raise RuntimeError('Precision-comparison source hash mismatch')
    smoke = '''from research_audit.compact_iqa import Images
import json, numpy as np
labels=json.load(open('dataset/train.json'))
a=Images('dataset/image',labels,['0000.tif'],512)[0][0].numpy()
b=Images('dataset/image',labels,['0000.tif'],512,quantize_uint8=True)[0][0].numpy()
assert np.isfinite(b).all() and b.min()>=0 and b.max()<=1
assert np.max(np.abs(a-b)) <= 1/255+1e-6
assert np.unique(b).size <= 256 and np.unique(a).size > 256
print('Precision control verified on one development training image',flush=True)
'''
    subprocess.run([sys.executable, '-c', smoke], cwd=work,
                   env={**os.environ, 'PYTHONPATH': str(root/'deps')+os.pathsep+str(work)}, check=True)

    def worker(gpu, seeds):
        env = {**os.environ, 'CUDA_VISIBLE_DEVICES': str(gpu),
               'PYTHONPATH': str(root/'deps')+os.pathsep+str(work)}
        status = {}
        for seed in seeds:
            common = ['research_audit.compact_iqa', '--batch-size', '32', '--workers', '4']
            jobs = [(f'precision_uint8_seed{seed}', common+[
                '--image-dir', 'dataset/image', '--labels', 'dataset/train.json',
                '--split', 'research_audit/development_split_seed42.json',
                '--pretrained-file', '../efficientnet_b0.safetensors', '--resolution', '448',
                '--epochs', '100', '--seed', str(seed), '--quantize-uint8',
                '--output', f'results/uint8_seed{seed}']),
                (f'eval_precision_uint8_seed{seed}', common+[
                    '--image-dir', '../LDCTIQAC_test/images', '--labels', '../LDCTIQAC_test/test.json',
                    '--checkpoint', f'results/uint8_seed{seed}/best.pth',
                    '--output', f'../evaluations/compact448_uint8_seed{seed}'])]
            for name, command in jobs:
                print(f'Starting {name} on GPU {gpu}', flush=True)
                started = time.monotonic()
                with (root/f'{name}.log').open('x') as log:
                    process = subprocess.run([sys.executable, '-u', '-m', *command], cwd=work,
                                             env=env, stdout=log, stderr=subprocess.STDOUT)
                status[name] = {'exit_code': process.returncode, 'wall_seconds': time.monotonic()-started,
                                'gpu': gpu, 'command': command}
                (root/f'precision_comparison_gpu{gpu}.json').write_text(json.dumps(status, indent=2))
                if process.returncode:
                    raise RuntimeError(f'{name} failed')
                print(f'Completed {name}', flush=True)
        return status
    with ThreadPoolExecutor(max_workers=2) as pool:
        jobs = [pool.submit(worker, 0, [42,456]), pool.submit(worker, 1, [123])]
        for job in jobs:
            job.result()


if __name__ == '__main__':
    main()
