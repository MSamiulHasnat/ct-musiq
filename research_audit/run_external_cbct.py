"""Run every declared model on the independent phantom data without selection."""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
import sys
import time


def main():
    root=Path.cwd(); store=root/'results/external_models'; status={}
    tasks=[]
    for arm in ['compact','rps','conditional','mild']:
        for seed in [42,123,456]:
            if arm=='compact':
                checkpoint=(root/f'results/compact448_seed{seed}/best.pth' if seed==42 else
                            store/f'development_screening/results/compact448_seed{seed}/best.pth')
            else:
                stage='mild_ordinal_study' if arm=='mild' else 'ordinal_study'
                checkpoint=store/f'{stage}/results/{arm}_seed{seed}/best.pth'
            output=root/f'research_audit/results/external_cbct_iq/{arm}_seed{seed}'
            if (output/'metrics.json').exists():
                status[f'{arm}_seed{seed}']={'status':'existing_completed_output'}
                continue
            if not checkpoint.is_file(): raise FileNotFoundError(checkpoint)
            tasks.append((arm,seed,checkpoint,output))
    def run(task):
        arm,seed,checkpoint,output=task; name=f'{arm}_seed{seed}'; started=time.monotonic()
        log=root/f'results/external_axial_{name}.log'
        print('Starting',name,flush=True)
        command=[sys.executable,'-u','-m','research_audit.evaluate_cbct_iq','--kind',
                 'compact' if arm=='compact' else 'ordinal','--checkpoint',str(checkpoint),
                 '--output',str(output),'--threads','4']
        with log.open('x') as stream:
            result=subprocess.run(command,cwd=root,stdout=stream,stderr=subprocess.STDOUT)
        record={'exit_code':result.returncode,'wall_seconds':time.monotonic()-started,'command':command}
        if result.returncode: raise RuntimeError(f'{name} failed; see {log}')
        print('Completed',name,flush=True)
        return name,record
    with ThreadPoolExecutor(max_workers=2) as pool:
        for name,record in pool.map(run,tasks):
            status[name]=record
            (root/'research_audit/results/external_cbct_iq/run_status.json').write_text(json.dumps(status,indent=2))


if __name__=='__main__': main()
