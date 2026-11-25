"""Fetch a known research artifact in bounded SSH chunks and verify SHA-256."""
import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
import hashlib
from pathlib import Path
import re
import subprocess


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--remote',required=True); p.add_argument('--output',required=True)
    p.add_argument('--size',type=int,required=True); p.add_argument('--sha256',required=True)
    args=p.parse_args()
    if not re.fullmatch(r'/[A-Za-z0-9_./-]+',args.remote): raise ValueError('Invalid remote research path')
    out=Path(args.output); parts=out.parent/(out.name+'.chunks'); parts.mkdir(parents=True,exist_ok=True)
    block=262144; total=(args.size+block-1)//block
    def fetch(index):
        expected=min(block,args.size-index*block); path=parts/f'{index:05d}'
        if path.exists() and path.stat().st_size==expected: return index
        command=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','-o','IPQoS=none',
                 'samiul@100.70.69.126',f'dd if={args.remote} bs={block} skip={index} count=1 status=none']
        for attempt in range(3):
            try:
                result=subprocess.run(command,capture_output=True,timeout=45,check=True)
                if len(result.stdout)!=expected: raise ValueError('Incomplete artifact chunk')
                path.write_bytes(result.stdout); return index
            except (subprocess.SubprocessError,ValueError):
                if attempt==2: raise
    with ThreadPoolExecutor(max_workers=4) as pool:
        jobs=[pool.submit(fetch,index) for index in range(total)]
        for done,job in enumerate(as_completed(jobs),1):
            job.result(); print(f'Fetched {done}/{total} chunks',flush=True)
    temporary=out.with_suffix(out.suffix+'.verified-tmp')
    digest=hashlib.sha256()
    with temporary.open('wb') as stream:
        for index in range(total):
            data=(parts/f'{index:05d}').read_bytes(); digest.update(data); stream.write(data)
    if digest.hexdigest()!=args.sha256: raise RuntimeError('Artifact SHA-256 mismatch')
    temporary.replace(out)
    print('Verified artifact',out,flush=True)


if __name__=='__main__': main()
