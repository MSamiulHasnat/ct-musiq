"""Finish the fixed strong-blur gate ablation and all external GPU controls.

This is an exploratory follow-up declared after the first external CPU results.
It does not tune a parameter or select a model using external labels.
"""
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
    root=Path(__file__).resolve().parent; work=root/'strong_unconditional_study'
    archive=root/'strong-unconditional-source.tar.gz'
    if hashlib.sha256(archive.read_bytes()).hexdigest()!=args.source_sha:
        raise RuntimeError('Frozen source archive mismatch')
    # Finish existing work before allocating additional training memory.
    required=[root/'evaluations'/f'ordinal_unconditional_seed{s}'/'metrics.json' for s in [42,123,456]]
    required += [root/'external_gpu_results'/f'{a}_seed{s}'/'metrics.json'
                 for a in ['compact','rps','conditional','mild'] for s in [42,123,456]]
    print('Waiting for current training and original external inference',flush=True)
    while not all(path.exists() for path in required): time.sleep(10)
    data=root/'external_gpu_study/dataset/external_cbct_iq/prepared_axial'

    def worker(gpu,seeds):
        env={**os.environ,'CUDA_VISIBLE_DEVICES':str(gpu),
             'PYTHONPATH':str(root/'deps')+os.pathsep+str(work)}
        status={}
        def run(name,module,options):
            start=time.monotonic(); print(f'Starting {name} on GPU {gpu}',flush=True)
            with (root/f'{name}.log').open('x') as stream:
                proc=subprocess.run([sys.executable,'-u','-m',module,*options],cwd=work,
                                    env=env,stdout=stream,stderr=subprocess.STDOUT)
            status[name]={'exit_code':proc.returncode,'wall_seconds':time.monotonic()-start,
                          'gpu':gpu,'module':module,'arguments':options}
            (root/f'final_controls_gpu{gpu}.json').write_text(json.dumps(status,indent=2))
            if proc.returncode: raise RuntimeError(f'{name} failed')
            print(f'Completed {name}',flush=True)
        for seed in seeds:
            name=f'strong_unconditional_seed{seed}'
            run(f'ordinal_{name}','research_audit.ordinal_iqa',[
                '--image-dir','dataset/image','--labels','dataset/train.json',
                '--split','research_audit/development_split_seed42.json',
                '--pretrained-file','../efficientnet_b0.safetensors','--output',f'results/{name}',
                '--seed',str(seed),'--degradation-weight','0.1','--constraint-sigma','1.5',
                '--head','ordinal','--quality-threshold','0'])
            run(f'probe_{name}','research_audit.degradation_probe',[
                '--kind','ordinal','--checkpoint',f'results/{name}/best.pth',
                '--image-dir','dataset/image','--labels','dataset/train.json',
                '--split','research_audit/development_split_seed42.json','--output',f'../probes/{name}'])
        return run
    with ThreadPoolExecutor(max_workers=2) as pool:
        jobs=[pool.submit(worker,0,[42,456]),pool.submit(worker,1,[123])]
        runners=[job.result() for job in jobs]
    for gpu,seeds in [(0,[42,456]),(1,[123])]:
        for seed in seeds:
            name=f'strong_unconditional_seed{seed}'
            runners[gpu](f'eval_ordinal_{name}','research_audit.ordinal_iqa',[
                '--image-dir','../LDCTIQAC_test/images','--labels','../LDCTIQAC_test/test.json',
                '--checkpoint',f'results/{name}/best.pth','--output',f'../evaluations/ordinal_{name}'])

    arms={'scalar_cond':'scalar_constraint_study','unconditional':'unconditional_study',
          'resnet_baseline':'resnet_constraint_study','resnet_cond':'resnet_constraint_study',
          'strong_unconditional':'strong_unconditional_study'}
    tasks=[(arm,stage,seed) for arm,stage in arms.items() for seed in [42,123,456]]
    def external(gpu,items):
        for arm,stage,seed in items:
            name=f'{arm}_seed{seed}'
            if not (root/'evaluations'/f'ordinal_{name}'/'metrics.json').exists():
                raise RuntimeError(f'Unfinished training/test control: {name}')
            runners[gpu](f'external_final_{name}','research_audit.evaluate_cbct_iq',[
                '--kind','ordinal' if arm=='strong_unconditional' else 'compact',
                '--checkpoint',str(root/stage/'results'/name/'best.pth'),
                '--data-root',str(data),'--device','cuda',
                '--output',str(root/'external_gpu_results'/name)])
    with ThreadPoolExecutor(max_workers=2) as pool:
        jobs=[pool.submit(external,gpu,tasks[gpu::2]) for gpu in [0,1]]
        for job in jobs: job.result()
    print('All final controls and 27-model external evaluation completed',flush=True)


if __name__=='__main__': main()
