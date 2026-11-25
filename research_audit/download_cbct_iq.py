"""Resume the public CBCT-IQ degraded archive on the LOCAL computer only."""
import hashlib
from pathlib import Path
import time
import urllib.request


def main():
    path=Path('dataset/external_cbct_iq/degraded.zip')
    url='https://zenodo.org/records/20124496/files/degraded.zip?download=1'
    size=777962838
    path.parent.mkdir(parents=True,exist_ok=True)
    offset=path.stat().st_size if path.exists() else 0
    if offset>size: raise ValueError('Local file exceeds published size')
    with path.open('ab') as stream:
        while offset<size:
            end=min(size-1,offset+8*1024*1024-1)
            for attempt in range(4):
                try:
                    request=urllib.request.Request(url,headers={'Range':f'bytes={offset}-{end}'})
                    with urllib.request.urlopen(request,timeout=35) as response:
                        if response.status!=206 or response.headers.get('Content-Range')!=f'bytes {offset}-{end}/{size}':
                            raise RuntimeError('Server did not honor the requested range')
                        block=response.read(end-offset+1)
                    if len(block)!=end-offset+1: raise RuntimeError('Incomplete download chunk')
                    stream.write(block); stream.flush(); offset=end+1
                    print(f'Downloaded {offset}/{size} bytes',flush=True)
                    break
                except Exception:
                    if attempt==3: raise
                    time.sleep(2)
    digest=hashlib.md5(path.read_bytes()).hexdigest()
    if digest!='3dc738f90301bcf1f87e59df69475e24': raise RuntimeError(f'MD5 mismatch: {digest}')
    print('Verified published archive MD5',digest,flush=True)


if __name__=='__main__': main()
