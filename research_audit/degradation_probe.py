"""Synthetic development stress tests; transformed images have no human labels."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision.transforms.functional import gaussian_blur
from research_audit.compact_iqa import CompactIQA,Images
from research_audit.ordinal_iqa import OrdinalIQA


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoint',required=True)
    p.add_argument('--kind',choices=['compact','ordinal'],required=True)
    p.add_argument('--image-dir',required=True)
    p.add_argument('--labels',required=True)
    p.add_argument('--split',required=True)
    p.add_argument('--output',required=True)
    args=p.parse_args()
    out=Path(args.output)
    if out.exists(): raise FileExistsError(out)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint=torch.load(args.checkpoint,map_location='cpu',weights_only=True)
    backbone=checkpoint['config'].get('backbone','efficientnet_b0')
    model=(CompactIQA(backbone=backbone) if args.kind=='compact' else OrdinalIQA(backbone=backbone)).to(device)
    model.load_state_dict(checkpoint['model_state_dict'],strict=True); model.eval()
    labels=json.loads(Path(args.labels).read_text())
    names=json.loads(Path(args.split).read_text())['val']
    loader=DataLoader(Images(args.image_dir,labels,names,448),batch_size=32,num_workers=4)
    variants=['gaussian_0.75','gaussian_1.5','gaussian_2.5','box_5','noise_0.01','noise_0.03']
    records=[]
    for images,targets,ids in loader:
        images=images.to(device)
        originals=model(images).flatten().cpu().numpy()
        for variant in variants:
            if variant.startswith('gaussian'):
                sigma=float(variant.split('_')[1]); kernel=2*int(np.ceil(3*sigma))+1
                altered=gaussian_blur(images,[kernel,kernel],[sigma,sigma])
            elif variant=='box_5':
                altered=F.avg_pool2d(F.pad(images,(2,2,2,2),mode='reflect'),5,1)
            else:
                sigma=float(variant.split('_')[1]); noise=[]
                for name in ids:
                    generator=torch.Generator().manual_seed(int.from_bytes(hashlib.sha256(name.encode()).digest()[:4],'little'))
                    noise.append(torch.randn((1,448,448),generator=generator).repeat(3,1,1))
                altered=(images+sigma*torch.stack(noise).to(device)).clamp(0,1)
            transformed=model(altered).flatten().cpu().numpy()
            records.extend({'image_id':name,'variant':variant,'original_target':float(target),
                            'original_prediction':float(a),'altered_prediction':float(b),'delta':float(b-a)}
                           for name,target,a,b in zip(ids,targets,originals,transformed))
    summary={}
    for variant in variants:
        summary[variant]={}
        for group in ['all','target_at_least_3','target_at_most_1']:
            chosen=[r for r in records if r['variant']==variant and
                    (group=='all' or group=='target_at_least_3' and r['original_target']>=3 or
                     group=='target_at_most_1' and r['original_target']<=1)]
            if chosen:
                delta=np.array([r['delta'] for r in chosen])
                summary[variant][group]={'n':len(delta),'mean_delta':float(delta.mean()),
                                         'fraction_score_increases':float((delta>0).mean())}
    out.mkdir(parents=True)
    with (out/'probe_predictions.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=records[0]); writer.writeheader(); writer.writerows(records)
    result={'warning':'Synthetic score changes only; no transformed-image MOS or clinical preference labels.',
            'checkpoint_sha256':hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest(),
            'split_sha256':hashlib.sha256(Path(args.split).read_bytes()).hexdigest(),
            'applied_at':'448x448 normalized float image after resizing','summary':summary}
    (out/'metrics.json').write_text(json.dumps(result,indent=2)); print(json.dumps(summary),flush=True)


if __name__=='__main__': main()
