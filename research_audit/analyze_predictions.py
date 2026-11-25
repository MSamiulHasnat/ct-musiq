"""Summarize prediction errors, including exploratory score-range summaries."""
import argparse
import csv
import json

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('predictions')
    args = parser.parse_args()
    with open(args.predictions, newline='', encoding='utf-8') as stream:
        rows = list(csv.DictReader(stream))
    pred = np.array([float(row['predicted']) for row in rows])
    target = np.array([float(row['target']) for row in rows])
    if len(pred) == 0 or not np.isfinite(pred).all() or not np.isfinite(target).all():
        raise ValueError('Predictions and targets must be non-empty and finite')
    errors = pred - target
    summary = {'n': len(rows), 'bias': float(errors.mean()),
               'mae': float(np.abs(errors).mean()),
               'rmse': float(np.sqrt(np.mean(errors ** 2))), 'by_target_range': {}}
    for low, high in [(0, 1), (1, 2), (2, 3), (3, 4.000001)]:
        mask = (target >= low) & (target < high)
        if mask.any():
            summary['by_target_range'][f'[{low}, {min(high, 4)}{ "]" if high > 4 else ")"}'] = {
                'n': int(mask.sum()), 'bias': float(errors[mask].mean()),
                'mae': float(np.abs(errors[mask]).mean())}
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
