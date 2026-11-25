"""Wait for two in-progress transfers, verify them, and run the audit once."""
import hashlib
import os
from pathlib import Path
import subprocess
import sys
import time
import zipfile

root = Path(__file__).resolve().parent
inputs = {
    root / 'LDCTIQAC_test.zip': (239751060, 'e017e8d555edb0193785f60efcdedce70f60cd2a2419deceb1aa5eda0582d4df'),
    root / 'ct_weights.pth': (353363861, 'a9cb39f582ce8b8aae0c7ffd098e1ca4fcd1c91655ec6112edf56c619ff05327'),
}
print('Waiting for transferred files', flush=True)
while not all(p.exists() and p.stat().st_size == size for p, (size, _) in inputs.items()):
    time.sleep(15)
# SFTP may issue writes out of order; allow in-flight writes to settle before
# checking hashes, rather than treating the file length alone as completion.
time.sleep(30)
for path, (_, expected) in inputs.items():
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    actual = digest.hexdigest()
    if actual != expected:
        raise RuntimeError(f'SHA-256 mismatch for {path}: {actual}')
    print(f'Verified {path.name}', flush=True)
with zipfile.ZipFile(root / 'LDCTIQAC_test.zip') as archive:
    for member in archive.infolist():
        if not (root / member.filename).resolve().is_relative_to(root):
            raise RuntimeError(f'Unsafe zip entry: {member.filename}')
    archive.extractall(root)
env = os.environ.copy()
env['PYTHONPATH'] = str(root / 'deps') + os.pathsep + str(root)
subprocess.run([sys.executable, str(root / 'evaluate_released_test.py'),
                '--data', str(root / 'LDCTIQAC_test'),
                '--checkpoint', str(root / 'ct_weights.pth'),
                '--output', str(root / 'exploratory_test')], cwd=root, env=env, check=True)
