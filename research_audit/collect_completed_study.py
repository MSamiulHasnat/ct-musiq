"""Export complete fixed-study evidence without images or model weights."""
import hashlib
import json
import platform
from pathlib import Path
import tarfile

import numpy
import scipy
import PIL
import timm
import torch
import torchvision
import nibabel


def main():
    root=Path(__file__).resolve().parent
    layout={
        'compact':('development_screening','screening'),
        'rps':('ordinal_study','ordinal_stage'),
        'conditional':('ordinal_study','ordinal_stage'),
        'mild':('mild_ordinal_study','mild_stage'),
        'scalar_cond':('scalar_constraint_study','scalar_stage'),
        'unconditional':('unconditional_study','unconditional_stage'),
        'resnet_baseline':('resnet_constraint_study','resnet_stage'),
        'resnet_cond':('resnet_constraint_study','resnet_stage'),
        'strong_unconditional':('strong_unconditional_study','strong_unconditional_stage')}
    files=[]; models=[]
    for arm,(study,stage) in layout.items():
        for seed in [42,123,456]:
            name=f'{arm}_seed{seed}'
            folder=f'compact448_seed{seed}' if arm=='compact' else name
            evaluation=(root/'evaluations'/folder if arm=='compact'
                        else root/'evaluations'/f'ordinal_{name}')
            destination=f'{stage}/evaluations/'+(folder if arm=='compact' else f'ordinal_{name}')
            for filename in ['metrics.json','predictions.csv']:
                files.append((evaluation/filename,f'{destination}/{filename}'))
            external=root/'external_gpu_results'/name
            for filename in ['metrics.json','predictions.csv','pair_predictions.csv']:
                files.append((external/filename,f'external_gpu/{name}/{filename}'))
            probe=root/'probes'/name
            files.append((probe/'metrics.json',f'{stage}/probes/{name}/metrics.json'))
            for path in probe.glob('*predictions.csv'):
                files.append((path,f'{stage}/probes/{name}/{path.name}'))
            training=root/study/'results'/folder
            for filename in ['training_log.csv','validation_predictions.csv']:
                files.append((training/filename,f'{stage}/training/{name}/{filename}'))
            missing=[str(src) for src,_ in files if not src.is_file()]
            if missing: raise RuntimeError(f'Incomplete declared study: {missing}')
            checkpoint=training/'best.pth'
            digest=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            test_record=json.loads((evaluation/'metrics.json').read_text())
            external_record=json.loads((external/'metrics.json').read_text())
            if 'checkpoint_sha256' in test_record and digest!=test_record['checkpoint_sha256']:
                raise RuntimeError(f'Checkpoint changed before LDCT evaluation: {name}')
            if digest!=external_record['checkpoint_sha256']:
                raise RuntimeError(f'Checkpoint changed between evaluations: {name}')
            if test_record['n']!=300 or external_record['overall']['n']!=1666:
                raise RuntimeError(f'Incomplete endpoints: {name}')
            models.append({'arm':arm,'seed':seed,'remote_checkpoint':str(checkpoint),
                'checkpoint_sha256':digest,'config':external_record['training_config']})
    manifest={'models':models,'n_models':len(models),'environment':{
        'python':platform.python_version(),'torch':str(torch.__version__),
        'torchvision':str(torchvision.__version__),'timm':timm.__version__,
        'numpy':numpy.__version__,'scipy':scipy.__version__,'Pillow':PIL.__version__,
        'nibabel':nibabel.__version__,'cuda':torch.version.cuda,
        'gpus':[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]},
        'warning':'Exploratory study; fixed public benchmark and two external phantoms.'}
    manifest_path=root/'completed_study_manifest.json'
    manifest_path.write_text(json.dumps(manifest,indent=2))
    files.append((manifest_path,'completed_study_manifest.json'))
    for pattern in ['final_controls_gpu*.json','resnet_constraint_gpu*.json','unconditional_gpu*.json']:
        for path in root.glob(pattern): files.append((path,f'run_status/{path.name}'))
    files.append((root/'external_engine_compatibility.json','external_gpu/external_engine_compatibility.json'))
    archive=root/'completed-robustness-study.tar.gz'
    with tarfile.open(archive,'x:gz') as stream:
        for source,destination in files: stream.add(source,arcname=destination)
    print(json.dumps({'archive':str(archive),'sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),
                      'n_models':len(models),'files':len(files)},indent=2))


if __name__=='__main__': main()
