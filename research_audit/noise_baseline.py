"""Scalar baseline using high-frequency variation and train-only calibration."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import laplace
from scipy.stats import kendalltau, pearsonr, spearmanr
from sklearn.isotonic import IsotonicRegression


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--images', required=True)
    parser.add_argument('--labels', required=True)
    parser.add_argument('--split', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--test-images', help='Optional separate test image directory')
    parser.add_argument('--test-labels', help='Optional separate test label JSON')
    args = parser.parse_args()
    if bool(args.test_images) != bool(args.test_labels):
        parser.error('--test-images and --test-labels must be provided together')
    labels = json.loads(Path(args.labels).read_text())
    split = json.loads(Path(args.split).read_text())
    if set(split['train']) & set(split['val']):
        raise ValueError('Train and validation images overlap')
    eval_labels = json.loads(Path(args.test_labels).read_text()) if args.test_labels else labels
    eval_names = sorted(eval_labels) if args.test_labels else split['val']
    eval_dir = Path(args.test_images) if args.test_images else Path(args.images)
    def energy(path):
        with Image.open(path) as image:
            pixels = np.clip(np.array(image, dtype=np.float32), 0, 1)
        return float(np.mean(laplace(pixels) ** 2))
    train_x = np.array([energy(Path(args.images) / name) for name in split['train']])
    train_y = np.array([labels[name] for name in split['train']])
    val_x = np.array([energy(eval_dir / name) for name in eval_names])
    val_y = np.array([eval_labels[name] for name in eval_names])
    direction = spearmanr(train_x, train_y).statistic >= 0
    reg = IsotonicRegression(increasing=direction, out_of_bounds='clip').fit(train_x, train_y)
    pred = reg.predict(val_x)
    p = pearsonr(pred, val_y).statistic
    s = spearmanr(pred, val_y).statistic
    k = kendalltau(pred, val_y).statistic
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    with (out / 'predictions.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['image_id', 'predicted', 'target', 'laplacian_energy'])
        writer.writerows(zip(eval_names, pred, val_y, val_x))
    result = {'n_train': len(train_y), 'n_evaluation': len(val_y),
              'evaluation_set': 'released_test' if args.test_labels else 'development_validation',
              'increasing': bool(direction),
              'PLCC': float(p), 'SROCC': float(s), 'KROCC': float(k),
              'Aggregate': float(abs(p) + abs(s) + abs(k)),
              'MAE': float(np.mean(np.abs(pred - val_y))),
              'warning': 'Exploratory analysis; inferred groups are not patient IDs.'}
    (out / 'metrics.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
