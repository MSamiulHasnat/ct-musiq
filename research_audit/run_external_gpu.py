"""Wait for verified local uploads, prepare fixed axial inputs, evaluate all models."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    root=Path(__file__).resolve().parent; work=root/'external_gpu_study'
    data=work/'dataset/external_cbct_iq'; archive=data/'degraded.zip'
    print('Waiting for complete locally downloaded external archive',flush=True)
    while not archive.exists() or archive.stat().st_size!=777962838: time.sleep(10)
    digest=hashlib.md5()
    with archive.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''): digest.update(block)
    if digest.hexdigest()!='3dc738f90301bcf1f87e59df69475e24': raise RuntimeError('External archive MD5 mismatch')
    env={**os.environ,'PYTHONPATH':str(root/'deps')+os.pathsep+str(work)}
    for module in ['research_audit.prepare_cbct_iq','research_audit.prepare_cbct_axial']:
        subprocess.run([sys.executable,'-u','-m',module],cwd=work,env=env,check=True)
    output=root/'external_gpu_results'; output.mkdir()

    def worker(gpu,arms):
        settings={**env,'CUDA_VISIBLE_DEVICES':str(gpu)}; status={}
        for arm in arms:
            for seed in [42,123,456]:
                name=f'{arm}_seed{seed}'
                stage='development_screening' if arm=='compact' else 'mild_ordinal_study' if arm=='mild' else 'ordinal_study'
                folder=f'compact448_seed{seed}' if arm=='compact' else name
                checkpoint=root/stage/'results'/folder/'best.pth'
                command=[sys.executable,'-u','-m','research_audit.evaluate_cbct_iq',
                    '--kind','compact' if arm=='compact' else 'ordinal','--checkpoint',str(checkpoint),
                    '--output',str(output/name),'--device','cuda']
                print(f'Starting {name} on GPU {gpu}',flush=True); started=time.monotonic()
                with (root/f'external_gpu_{name}.log').open('x') as log:
                    proc=subprocess.run(command,cwd=work,env=settings,stdout=log,stderr=subprocess.STDOUT)
                status[name]={'exit_code':proc.returncode,'wall_seconds':time.monotonic()-started,
                              'gpu':gpu,'command':command}
                (output/f'gpu{gpu}_status.json').write_text(json.dumps(status,indent=2))
                if proc.returncode: raise RuntimeError(f'{name} failed')
                print(f'Completed {name}',flush=True)
    with ThreadPoolExecutor(max_workers=2) as pool:
        jobs=[pool.submit(worker,0,['compact','rps']),pool.submit(worker,1,['conditional','mild'])]
        for job in jobs: job.result()


if __name__=='__main__': main()
