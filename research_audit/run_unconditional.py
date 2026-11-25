"""Execute two declared arms on two GPUs, retaining every seeded outcome."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    p=argparse.ArgumentParser(); p.add_argument('--source-sha',required=True); args=p.parse_args()
    root=Path(__file__).resolve().parent; work=root/'unconditional_study'
    if hashlib.sha256((root/'unconditional-source.tar.gz').read_bytes()).hexdigest()!=args.source_sha:
        raise RuntimeError('Study source archive hash mismatch')

    def worker(gpu,arm,weight):
        env={**os.environ,'CUDA_VISIBLE_DEVICES':str(gpu),'PYTHONPATH':str(root/'deps')+os.pathsep+str(work)}
        status={}
        def run(name,module,options):
            start=time.monotonic()
            print(f'Starting {name} on GPU {gpu}',flush=True)
            with (root/f'{name}.log').open('x') as log:
                proc=subprocess.run([sys.executable,'-u','-m',module,*options],cwd=work,env=env,
                                    stdout=log,stderr=subprocess.STDOUT)
            status[name]={'exit_code':proc.returncode,'wall_seconds':time.monotonic()-start,
                          'gpu':gpu,'module':module,'arguments':options}
            (root/f'unconditional_gpu{gpu}.json').write_text(json.dumps(status,indent=2))
            if proc.returncode: raise RuntimeError(f'{name} failed')
            print(f'Completed {name}',flush=True)
        for seed in ([42,456] if gpu==0 else [123]):
            name=f'{arm}_seed{seed}'
            run(f'ordinal_{name}','research_audit.ordinal_iqa',[
                '--image-dir','dataset/image','--labels','dataset/train.json',
                '--split','research_audit/development_split_seed42.json',
                '--pretrained-file','../efficientnet_b0.safetensors','--output',f'results/{name}',
                '--seed',str(seed),'--degradation-weight',str(weight),'--constraint-sigma','0.75','--head','scalar','--quality-threshold','0'])
            run(f'probe_{name}','research_audit.degradation_probe',[
                '--kind','compact','--checkpoint',f'results/{name}/best.pth',
                '--image-dir','dataset/image','--labels','dataset/train.json',
                '--split','research_audit/development_split_seed42.json','--output',f'../probes/{name}'])
        return run
    with ThreadPoolExecutor(max_workers=2) as pool:
        jobs=[pool.submit(worker,0,'unconditional',.1),pool.submit(worker,1,'unconditional',.1)]
        runs=[job.result() for job in jobs]
    # All development work finishes before any new test evaluation starts.
    for (arm,run) in zip(['unconditional','unconditional'],runs):
        for seed in ([42,456] if run is runs[0] else [123]):
            name=f'{arm}_seed{seed}'
            run(f'eval_ordinal_{name}','research_audit.ordinal_iqa',[
                '--image-dir','../LDCTIQAC_test/images','--labels','../LDCTIQAC_test/test.json',
                '--checkpoint',f'results/{name}/best.pth','--output',f'../evaluations/ordinal_{name}'])


if __name__=='__main__': main()
