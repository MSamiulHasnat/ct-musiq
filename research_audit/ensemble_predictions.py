"""Evaluate one fixed equal-weight average of saved model predictions."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr, kendalltau


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--inputs', nargs='+', required=True)
    parser.add_argument('--expected-n', type=int, required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    arrays, source_metrics, sources = [], [], []
    reference_names = reference_target = None

    def metrics(pred, target):
        values = [float(fn(pred, target).statistic) for fn in (pearsonr, spearmanr, kendalltau)]
        return dict(zip(['PLCC', 'SROCC', 'KROCC'], values),
                    Aggregate=sum(abs(x) for x in values), mae=float(np.abs(pred-target).mean()),
                    bias=float((pred-target).mean()))

    for name in args.inputs:
        path = Path(name)
        with path.open(newline='') as stream:
            rows = sorted(csv.DictReader(stream), key=lambda row: row['image_id'])
        names = [row['image_id'] for row in rows]
        pred = np.array([float(row['predicted']) for row in rows])
        target = np.array([float(row['target']) for row in rows])
        if len(names) != args.expected_n or len(set(names)) != len(names) or not np.isfinite([pred,target]).all():
            raise ValueError(f'Invalid prediction set: {path}')
        if reference_names is None:
            reference_names, reference_target = names, target
        if names != reference_names or not np.allclose(target, reference_target, rtol=0, atol=1e-6):
            raise ValueError('Input files disagree on image IDs or target scores')
        arrays.append(pred)
        source_metrics.append(metrics(pred,target))
        sources.append({'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
    average = np.mean(arrays, axis=0)
    result = {'rule': 'equal arithmetic mean; no fitted weights or calibration',
              'n': len(reference_names), 'models': len(arrays), 'sources': sources,
              'individual_metrics': source_metrics, 'ensemble_metrics': metrics(average, reference_target),
              'warning': 'Exploratory analysis after prior public test use; inferred development groups are not patient IDs.'}
    output.mkdir(parents=True)
    with (output/'predictions.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['image_id','predicted','target'])
        writer.writerows(zip(reference_names,average,reference_target))
    (output/'metrics.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result['ensemble_metrics'],indent=2))


if __name__ == '__main__':
    main()
