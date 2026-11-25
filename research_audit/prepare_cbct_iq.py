"""Prepare all expert-annotated CBCT-IQ slices without fitting to their ratings."""
import csv
import gzip
import hashlib
import json
from pathlib import Path
import zipfile

import nibabel as nib
import numpy as np
from PIL import Image


def main():
    root=Path('dataset/external_cbct_iq')
    source=root/'degraded.zip'; annotation=root/'annotations.csv'; output=root/'prepared'
    for path,expected in [(source,'3dc738f90301bcf1f87e59df69475e24'),
                          (annotation,'37733870d1f6cb796272b5f778a98c0c')]:
        digest=hashlib.md5()
        with path.open('rb') as stream:
            for block in iter(lambda:stream.read(1024*1024),b''): digest.update(block)
        if digest.hexdigest()!=expected: raise ValueError(f'Published MD5 mismatch: {path}')
    rows=list(csv.DictReader(annotation.open(newline='')))
    if len(rows)!=1666 or len({r['filename'] for r in rows})!=1666:
        raise ValueError('Unexpected annotations or duplicate image paths')
    expected_files={f'images/cbct{i:04d}.tiff' for i in range(len(rows))}|{'labels.json','metadata.csv','manifest.json'}
    if output.exists() and any(p.relative_to(output).as_posix() not in expected_files for p in output.rglob('*') if p.is_file()):
        raise ValueError('Unexpected files in prepared-data directory')
    output.mkdir(exist_ok=True); (output/'images').mkdir(exist_ok=True)
    labels,metadata={},[]
    with zipfile.ZipFile(source) as archive:
        names=archive.namelist()
        matched=[]
        for row in rows:
            fields=row['filename'].removeprefix('./').replace('\\','/').split('/')
            filename='VERYSMOOTH.nii.gz' if fields[-1]=='VSMOOTH.nii.gz' else fields[-1]
            key='/'.join([fields[0],fields[2],fields[1],*fields[3:-1],filename])
            candidates=[n for n in names if n==key or n.endswith('/'+key)]
            if len(candidates)!=1: raise ValueError(f'Missing/ambiguous annotated image: {key}')
            matched.append(candidates[0])
        for index,row in enumerate(rows):
            relative=row['filename'].removeprefix('./').replace('\\','/')
            # Published CSV is anatomy/slice/kernel; archive is anatomy/kernel/slice.
            fields=relative.split('/')
            image=nib.Nifti1Image.from_bytes(gzip.decompress(archive.read(matched[index])))
            pixels=np.asarray(image.dataobj,dtype=np.float32).squeeze()
            if pixels.ndim!=2 or not np.isfinite(pixels).all():
                raise ValueError(f'Expected one finite 2D slice: {relative}, {pixels.shape}')
            # Keep stored array orientation and published reference normalization.
            # The model loader clips to [0,1] and resizes identically to LDCT.
            name=f'cbct{index:04d}.tiff'
            destination=output/'images'/name
            if destination.exists():
                with Image.open(destination) as existing:
                    if not np.array_equal(np.asarray(existing),pixels):
                        raise ValueError(f'Existing prepared pixels disagree: {name}')
            else: Image.fromarray(pixels,mode='F').save(destination,compression='tiff_adobe_deflate')
            target=float(row['OIQ_MOS_ZSCORE'])
            if not np.isfinite(target): raise ValueError('Missing expert OIQ score')
            labels[name]=target
            parts=relative.split('/')
            metadata.append({'image_id':name,'source_path':relative,'anatomy':parts[0],
                'slice':parts[1],'reconstruction':parts[-1].removesuffix('.nii.gz'),
                'target':target,'reader_mean':sum(float(row[f'OIQ_{i}']) for i in [1,2,3])/3,
                'minimum':float(pixels.min()),'maximum':float(pixels.max()),
                'fraction_clipped':float(((pixels<0)|(pixels>1)).mean()),
                'nifti_axis_codes':str(nib.aff2axcodes(image.affine))})
            if index%100==0: print(f'Prepared {index+1}/{len(rows)} annotated images',flush=True)
    (output/'labels.json').write_text(json.dumps(labels,indent=2))
    with (output/'metadata.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=metadata[0]); writer.writeheader(); writer.writerows(metadata)
    manifest={'source':'https://zenodo.org/records/20124496','license':'CC BY 4.0','n':len(rows),
        'target':'Human OIQ_MOS_ZSCORE, higher is better; no algorithm-derived IQA ratings',
        'orientation':'Stored 2D NIfTI array, unchanged; no orientation search',
        'path_mapping':'CSV anatomy/slice/kernel -> archive anatomy/kernel/slice; VSMOOTH -> VERYSMOOTH; unique exact match',
        'normalization':'Published reference normalization retained; common model loader clips [0,1]',
        'warnings':['Two phantoms, not patients.','External labels do not select or fit any model.',
                    'No score calibration across different rating scales; use rank correlation.'],
        'mean_fraction_clipped':float(np.mean([r['fraction_clipped'] for r in metadata]))}
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2)); print(json.dumps(manifest),flush=True)


if __name__=='__main__': main()
