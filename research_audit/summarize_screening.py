"""Recompute full-precision descriptive metrics from saved screening predictions."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr, kendalltau


def read(path):
    with path.open(newline='') as stream:
        rows = sorted(csv.DictReader(stream), key=lambda row: row['image_id'])
    names = [row['image_id'] for row in rows]
    pred = np.array([float(row['predicted']) for row in rows])
    target = np.array([float(row['target']) for row in rows])
    if len(names) != 300 or len(set(names)) != 300 or not np.isfinite([pred, target]).all():
        raise ValueError(f'Expected 300 unique, finite predictions: {path}')
    correlations = [float(fn(pred, target).statistic) for fn in (pearsonr, spearmanr, kendalltau)]
    metrics = dict(zip(['PLCC', 'SROCC', 'KROCC'], correlations))
    metrics.update(Aggregate=sum(abs(value) for value in correlations),
                   mae=float(np.abs(pred-target).mean()), bias=float((pred-target).mean()),
                   rmse=float(np.sqrt(np.square(pred-target).mean())))
    return names, target, metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results-root', default='research_audit/results')
    args = parser.parse_args()
    root = Path(args.results_root)
    result = {'description': 'One exploratory released test collection; sample SD across training seeds.',
              'limitations': ['Development groups are inferred, not verified patient IDs.',
                             'The released test has prior exploratory use.',
                             'Model pipelines differ in multiple settings; no isolated component effect.',
                             'Seed SD is not patient-level uncertainty.'], 'models': {}}
    canonical = None
    models = ['compact448', 'legacy_batch32', 'aligned']
    if (root/'precision_control').exists():
        models.append('compact448_uint8')
    for model in models:
        rows = []
        for seed in ([42] if model == 'aligned' else [42, 123, 456]):
            directory = (root/'precision_control'/'evaluations'/f'compact448_uint8_seed{seed}'
                         if model == 'compact448_uint8' else
                         root/'remote_stage2'/'evaluations'/f'batch32_seed{seed}'
                         if model == 'legacy_batch32' else
                         root/'screening'/'evaluations'/f'{model}_seed{seed}')
            names, targets, metrics = read(directory/'predictions.csv')
            if canonical is None:
                canonical = names, targets
            if names != canonical[0] or not np.allclose(targets, canonical[1], rtol=0, atol=1e-6):
                raise ValueError('Prediction sets disagree on image IDs or targets')
            recorded = json.loads((directory/'metrics.json').read_text())
            for key in ['PLCC', 'SROCC', 'KROCC', 'Aggregate']:
                if abs(metrics[key]-recorded[key]) > 0.000051:
                    raise ValueError(f'Recomputed {key} disagrees with saved output: {directory}')
            rows.append({'seed': seed, **metrics})
        summary = {key: {'mean': float(np.mean([row[key] for row in rows])),
                         'sample_sd': float(np.std([row[key] for row in rows], ddof=1)) if len(rows)>1 else None}
                   for key in rows[0] if key != 'seed'}
        result['models'][model] = {'n_images': 300, 'runs': rows, 'summary': summary}
    if 'compact448_uint8' in result['models']:
        first = result['models']['compact448']['runs']
        second = result['models']['compact448_uint8']['runs']
        differences = [{key: a[key]-b[key] for key in a if key != 'seed'}
                       for a,b in zip(first,second)]
        result['precision_effect_float_minus_uint8'] = {
            key: {'mean': float(np.mean([row[key] for row in differences])),
                  'sample_sd': float(np.std([row[key] for row in differences], ddof=1))}
            for key in differences[0]}
    output = root/'screening'/'summary.json'
    output.write_text(json.dumps(result, indent=2))
    print(json.dumps({name: item['summary'] for name, item in result['models'].items()}, indent=2))


if __name__ == '__main__':
    main()
