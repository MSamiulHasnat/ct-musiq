"""Map known RAS NIfTI voxel axes to axial image rows/columns, without score tuning."""
import csv
import json
from pathlib import Path
import shutil
import numpy as np
from PIL import Image


def main():
    root=Path('dataset/external_cbct_iq'); source=root/'prepared'; out=root/'prepared_axial'
    if out.exists(): raise FileExistsError(out)
    rows=list(csv.DictReader((source/'metadata.csv').open(newline='')))
    if len(rows)!=1666 or {r['nifti_axis_codes'] for r in rows}!={"('R', 'A', 'S')"}:
        raise ValueError('Unexpected image geometry')
    out.mkdir(); (out/'images').mkdir()
    for i,row in enumerate(rows):
        with Image.open(source/'images'/row['image_id']) as image:
            pixels=np.asarray(image,dtype=np.float32)
        # RAS x is the column direction; anterior (+y) appears at the top.
        axial=np.flipud(pixels.T).copy()
        Image.fromarray(axial,mode='F').save(out/'images'/row['image_id'],compression='tiff_adobe_deflate')
        if i%200==0: print(f'Mapped axial geometry {i+1}/{len(rows)}',flush=True)
    for name in ['labels.json','metadata.csv']: shutil.copyfile(source/name,out/name)
    manifest=json.loads((source/'manifest.json').read_text())
    manifest['orientation']='Known RAS NIfTI x/y axes mapped to axial rows/columns with flipud(array.T); anterior at image top. Fixed from geometry and input visual QA, no orientation-score search.'
    manifest['history']='Initial stored-array evaluations used a rotated presentation and were stopped and preserved separately; every model is rerun with this geometry correction.'
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))


if __name__=='__main__': main()
