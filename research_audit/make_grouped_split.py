"""Create a reproducible development split from provisional slice groups."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--groups', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    with Path(args.groups).open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 1000 or len({r['filename'] for r in rows}) != 1000:
        raise ValueError('Expected 1,000 unique image names')
    groups = sorted({int(r['inferred_slice_group']) for r in rows})
    rng = np.random.default_rng(args.seed)
    val_groups = set(rng.permutation(groups)[:round(len(groups) * 0.2)].tolist())
    result = {'method': 'inferred slice groups; patient identity unverified',
              'seed': args.seed,
              'train': [r['filename'] for r in rows if int(r['inferred_slice_group']) not in val_groups],
              'val': [r['filename'] for r in rows if int(r['inferred_slice_group']) in val_groups]}
    Path(args.output).write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(f"Training: {len(result['train'])} images; validation: {len(result['val'])} images")
    print('Warning: these inferred groups do not establish patient separation.')


if __name__ == '__main__':
    main()
