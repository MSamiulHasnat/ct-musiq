"""Exploratory bootstrap over inferred test-image groups, not patient IDs."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.stats import kendalltau, pearsonr, spearmanr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--images', required=True)
    parser.add_argument('--predictions', required=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--samples', type=int, default=2000)
    parser.add_argument('--comparison', help='Optional paired prediction CSV for difference intervals')
    args = parser.parse_args()
    with open(args.predictions, newline='', encoding='utf-8') as stream:
        rows = sorted(csv.DictReader(stream), key=lambda row: row['image_id'])
    features = []
    for row in rows:
        with Image.open(Path(args.images) / row['image_id']) as image:
            pixels = np.array(image.resize((64, 64)), dtype=np.float32).ravel()
        pixels -= pixels.mean()
        norm = np.linalg.norm(pixels)
        if norm == 0:
            raise ValueError(f"Constant image: {row['image_id']}")
        features.append(pixels / norm)
    features = np.stack(features)
    count, groups = connected_components(csr_matrix(features @ features.T > 0.99955), directed=False)
    pred = np.array([float(row['predicted']) for row in rows])
    target = np.array([float(row['target']) for row in rows])
    other = None
    if args.comparison:
        with open(args.comparison, newline='', encoding='utf-8') as stream:
            comparison = {row['image_id']: row for row in csv.DictReader(stream)}
        if set(comparison) != {row['image_id'] for row in rows}:
            raise ValueError('Prediction files must contain the same unique image IDs')
        if any(abs(float(comparison[row['image_id']]['target']) - float(row['target'])) > 1e-6 for row in rows):
            raise ValueError('Prediction files disagree on target scores')
        other = np.array([float(comparison[row['image_id']]['predicted']) for row in rows])
    rng = np.random.default_rng(args.seed)
    samples = []
    for _ in range(args.samples):
        selected = rng.choice(count, count, replace=True)
        indices = np.concatenate([np.flatnonzero(groups == group) for group in selected])
        x, y = pred[indices], target[indices]
        p, s, k = pearsonr(x, y).statistic, spearmanr(x, y).statistic, kendalltau(x, y).statistic
        values = [p, s, k, abs(p) + abs(s) + abs(k), np.abs(x-y).mean()]
        if other is not None:
            q = other[indices]
            other_score = sum(abs(fn(q, y).statistic) for fn in (pearsonr, spearmanr, kendalltau))
            values.extend([values[3] - other_score, values[4] - np.abs(q-y).mean()])
        samples.append(values)
    intervals = np.quantile(np.array(samples), [0.025, 0.5, 0.975], axis=0).T
    names = ['pearson', 'spearman', 'kendall', 'aggregate', 'mae']
    if other is not None:
        names += ['aggregate_difference_first_minus_comparison', 'mae_difference_first_minus_comparison']
    result = {'seed': args.seed, 'bootstrap_samples': args.samples,
              'inferred_groups': count, 'threshold': 0.99955,
              'warning': 'Image groups are inferred; patient independence is not established.',
              'percentiles_2_5_50_97_5': dict(zip(names, intervals.tolist()))}
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
