"""Paired, anatomy-stratified source-slice resampling within two fixed phantoms.

Intervals describe the observed phantom experiment, not patient/site populations.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--results',required=True); p.add_argument('--samples',type=int,default=2000)
    p.add_argument('--metadata',default='dataset/external_cbct_iq/prepared_axial/metadata.csv')
    p.add_argument('--require-complete',action='store_true')
    args=p.parse_args(); root=Path(args.results)
    metadata={r['image_id']:r for r in csv.DictReader(Path(args.metadata).open(newline=''))}
    candidates=['compact','rps','conditional','mild','scalar_cond','unconditional',
                'resnet_baseline','resnet_cond','strong_unconditional']
    models={}; target=names=pair_keys=None; pair_groups=None
    for arm in candidates:
        files=[root/f'{arm}_seed{s}' for s in [42,123,456]]
        if not all((f/'metrics.json').exists() for f in files): continue
        predictions,credits=[],[]
        for directory in files:
            rows=sorted(csv.DictReader((directory/'predictions.csv').open(newline='')),key=lambda r:r['image_id'])
            ids=[r['image_id'] for r in rows]; y=np.array([float(r['target']) for r in rows])
            x=np.array([float(r['predicted']) for r in rows])
            if len(ids)!=1666 or len(set(ids))!=1666 or not np.isfinite([x,y]).all(): raise ValueError('Invalid external predictions')
            if names is None: names,target=ids,y
            if ids!=names or not np.allclose(y,target,atol=1e-6,rtol=0): raise ValueError('Unpaired external images or targets')
            pairs=sorted(csv.DictReader((directory/'pair_predictions.csv').open(newline='')),key=lambda r:(r['first'],r['second']))
            keys=[(r['first'],r['second']) for r in pairs]
            if pair_keys is None:
                pair_keys=keys; pair_groups=['/'.join(r['group'].split('/')[:2]) for r in pairs]
            if keys!=pair_keys: raise ValueError('Unpaired expert reconstruction comparisons')
            predictions.append(x); credits.append([float(r['agreement_credit']) for r in pairs])
        models[arm]=(np.array(predictions),np.array(credits))
    if not models: raise ValueError('No complete three-seed model set')
    if args.require_complete and set(models)!=set(candidates):
        raise ValueError('All nine declared arms are required for final analysis')
    image_groups=[metadata[name]['anatomy']+'/'+metadata[name]['slice'] for name in names]
    groups=sorted(set(image_groups)); by_anatomy={a:[g for g in groups if g.startswith(a+'/')]
        for a in sorted({g.split('/')[0] for g in groups})}
    image_indices={g:np.flatnonzero(np.array(image_groups)==g) for g in groups}
    pair_indices={g:np.flatnonzero(np.array(pair_groups)==g) for g in groups}
    contrasts=[(a,b) for a,b in [('conditional','rps'),('mild','rps'),('conditional','compact'),
        ('mild','compact'),('scalar_cond','compact'),('scalar_cond','unconditional'),
        ('resnet_cond','resnet_baseline'),('conditional','strong_unconditional')] if a in models and b in models]
    def statistics(images,pairs):
        return {a:np.array([np.mean([spearmanr(x[images],target[images]).statistic for x in values[0]]),
                           values[1][:,pairs].mean()]) for a,values in models.items()}
    point=statistics(np.arange(len(names)),np.arange(len(pair_keys)))
    samples={f'{a}_minus_{b}':[] for a,b in contrasts}; rng=np.random.default_rng(42)
    for iteration in range(args.samples):
        chosen=[g for pool in by_anatomy.values() for g in rng.choice(pool,len(pool),replace=True)]
        values=statistics(np.concatenate([image_indices[g] for g in chosen]),
                          np.concatenate([pair_indices[g] for g in chosen]))
        for a,b in contrasts: samples[f'{a}_minus_{b}'].append(values[a]-values[b])
        if iteration%250==0: print(f'Completed {iteration}/{args.samples} resamples',flush=True)
    result={'samples':args.samples,'seed':42,'source_slice_groups':len(groups),
        'groups_per_anatomy':{a:len(g) for a,g in by_anatomy.items()},'n_images':len(names),'n_pairs':len(pair_keys),
        'warning':'Conditional source-slice intervals within two fixed phantoms and three fixed training seeds; not patient/site population confidence intervals.',
        'comparisons':{}}
    for a,b in contrasts:
        key=f'{a}_minus_{b}'; interval=np.quantile(np.array(samples[key]),[.025,.5,.975],axis=0)
        result['comparisons'][key]={name:{'point':float((point[a]-point[b])[i]),
            'percentiles_2_5_50_97_5':interval[:,i].tolist()} for i,name in enumerate(['SROCC','pair_agreement'])}
    (root/'paired_source_slice_bootstrap.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result['comparisons'],indent=2),flush=True)


if __name__=='__main__': main()
