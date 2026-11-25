"""Verify offline inputs and start the provisional grouped development run."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import zipfile

root = Path(__file__).resolve().parent
inputs = {
    root / 'ct-musiq-train-data.zip': (859138106, '44ea6b1acbd285e64a8033026863be08f7847138f8ac48b1fd3b125dd8900089'),
    root / 'vit_base_patch32_224.safetensors': (352911016, 'b0a4866d4f722434e9d26c529181f1e7072d80edcf75843f2af4e091617a4098'),
}
print('Waiting for offline training inputs', flush=True)
while not all(path.exists() and path.stat().st_size == size for path, (size, _) in inputs.items()):
    time.sleep(15)
time.sleep(30)
for path, (_, expected) in inputs.items():
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    if digest.hexdigest() != expected:
        raise RuntimeError(f'SHA-256 mismatch for {path}')
    print(f'Verified {path.name}', flush=True)
work = root / 'grouped'
with zipfile.ZipFile(root / 'ct-musiq-train-data.zip') as archive:
    for member in archive.infolist():
        if not (work / member.filename).resolve().is_relative_to(work.resolve()):
            raise RuntimeError(f'Unsafe archive entry: {member.filename}')
    archive.extractall(work)
jobs = []
for gpu, batch in [(0, 32), (1, 64)]:
    env = os.environ.copy()
    env['PYTHONPATH'] = str(root / 'deps') + os.pathsep + str(work)
    env['CUDA_VISIBLE_DEVICES'] = str(gpu)
    log = (root / f'grouped_batch{batch}.log').open('w')
    command = [sys.executable, '-u', '-m', 'ct_musiq.train',
               '--epochs', '100', '--batch_size', str(batch), '--lambda_kl', '0.10',
               '--split-manifest', 'research_audit/development_split_seed42.json',
               '--output-dir', f'results/development_seed42_batch{batch}', '--seed', '42',
               '--num-workers', '4', '--pretrained-file', str(root / 'vit_base_patch32_224.safetensors')]
    proc = subprocess.Popen(command, cwd=work, env=env, stdout=log, stderr=subprocess.STDOUT)
    jobs.append((gpu, batch, proc, log))
    print(f'Started GPU {gpu}, batch {batch}, PID {proc.pid}', flush=True)
results = {}
for gpu, batch, proc, log in jobs:
    results[f'batch{batch}'] = {'gpu': gpu, 'exit_code': proc.wait()}
    log.close()
    print(f'GPU {gpu}, batch {batch} exited {results[f"batch{batch}"]["exit_code"]}', flush=True)
(root / 'grouped_run_status.json').write_text(json.dumps(results, indent=2) + '\n')
if any(item['exit_code'] for item in results.values()):
    raise SystemExit(1)
