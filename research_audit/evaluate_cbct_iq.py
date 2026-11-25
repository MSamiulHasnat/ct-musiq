"""External phantom transfer: fixed inference, human-score rank correlation."""
import argparse
import csv
import hashlib
from itertools import combinations
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from scipy.stats import pearsonr,spearmanr,kendalltau
import timm
from research_audit.compact_iqa import CompactIQA,Images,predict,save_predictions
from research_audit.ordinal_iqa import OrdinalIQA


def main():
    p=argparse.ArgumentParser(); p.add_argument('--checkpoint',required=True)
    p.add_argument('--kind',choices=['compact','ordinal'],required=True)
    p.add_argument('--output',required=True); p.add_argument('--threads',type=int,default=4)
    p.add_argument('--data-root',default='dataset/external_cbct_iq/prepared_axial')
    p.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    args=p.parse_args(); torch.set_num_threads(args.threads)
    device=torch.device(args.device)
    root=Path(args.data_root); out=Path(args.output)
    if out.exists(): raise FileExistsError(out)
    checkpoint=torch.load(args.checkpoint,map_location='cpu',weights_only=True)
    backbone=checkpoint['config'].get('backbone','efficientnet_b0')
    model=CompactIQA(backbone=backbone) if args.kind=='compact' else OrdinalIQA(backbone=backbone)
    model.load_state_dict(checkpoint['model_state_dict'],strict=True); model.eval()
    model.to(device=device,memory_format=torch.channels_last)
    labels=json.loads((root/'labels.json').read_text())
    rows=list(csv.DictReader((root/'metadata.csv').open(newline='')))
    metadata={r['image_id']:r for r in rows}
    if len(labels)!=1666 or set(metadata)!=set(labels): raise ValueError('Incomplete external cases')
    names=sorted(labels); predictions=[]; targets=[]
    loader=DataLoader(Images(root/'images',labels,names,448),batch_size=8,num_workers=0)
    with torch.inference_mode():
        for index,(images,scores,_) in enumerate(loader):
            predictions.extend(model(images.contiguous(memory_format=torch.channels_last).to(device)).flatten().cpu().tolist()); targets.extend(scores.tolist())
            if index%25==0: print(f'Inferred {len(predictions)}/{len(names)} images',flush=True)
    pred,target=np.array(predictions),np.array(targets)
    def summarize(indices):
        x,y=pred[indices],target[indices]
        return {'n':len(x),'SROCC':float(spearmanr(x,y).statistic),
                'PLCC':float(pearsonr(x,y).statistic),'KROCC':float(kendalltau(x,y).statistic)}
    result={'overall':summarize(np.arange(len(names))), 'by_anatomy':{},
        'checkpoint_sha256':hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest(),
        'target':'Human OIQ_MOS_ZSCORE, signed rank correlation; no calibration',
        'training_config':checkpoint['config'],'torch':str(torch.__version__),'timm':timm.__version__,
        'device':str(device),'gpu':torch.cuda.get_device_name() if device.type=='cuda' else None,
        'data_manifest':json.loads((root/'manifest.json').read_text()),
        'warnings':['Two phantoms; images are correlated, not independent patients.',
            'Different modality, anatomy and score scale; no cross-scale MAE is reported.',
            'No external fitting or configuration selection is permitted.']}
    for anatomy in sorted({r['anatomy'] for r in rows}):
        result['by_anatomy'][anatomy]=summarize(np.array([i for i,n in enumerate(names) if metadata[n]['anatomy']==anatomy]))
    ratings={r['filename'].removeprefix('./'): [float(r[f'OIQ_{i}']) for i in [1,2,3]]
             for r in csv.DictReader((root.parent/'annotations.csv').open(newline=''))}
    groups={}
    for i,name in enumerate(names):
        key=metadata[name]['source_path'].rsplit('/',1)[0]
        groups.setdefault(key,[]).append(i)
    pairs=[]; ambiguous=0
    for key,indices in groups.items():
        for i,j in combinations(indices,2):
            votes=np.sign(np.array(ratings[metadata[names[i]]['source_path']])-np.array(ratings[metadata[names[j]]['source_path']]))
            direction=1 if (votes>0).sum()>=2 else -1 if (votes<0).sum()>=2 else 0
            if not direction: ambiguous+=1; continue
            difference=float(pred[i]-pred[j])
            credit=.5 if abs(difference)<=1e-6 else float(np.sign(difference)==direction)
            pairs.append({'first':names[i],'second':names[j],'group':key,'expert_majority':direction,
                          'prediction_difference':difference,'agreement_credit':credit})
    result['reconstruction_pair_agreement']={'n':len(pairs),'ambiguous_or_tied_expert_pairs':ambiguous,
        'agreement':float(np.mean([r['agreement_credit'] for r in pairs])) if pairs else None,
        'definition':'Same anatomy, slice, kernel, kV and pulse; different reconstruction; at least 2/3 readers agree on direction; model ties credit 0.5.'}
    out.mkdir(parents=True)
    save_predictions(out/'predictions.csv',names,pred,target)
    if pairs:
        with (out/'pair_predictions.csv').open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=pairs[0]); writer.writeheader(); writer.writerows(pairs)
    (out/'metrics.json').write_text(json.dumps(result,indent=2)); print(json.dumps(result),flush=True)


if __name__=='__main__': main()
