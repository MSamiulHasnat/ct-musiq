"""Suggest conservative near-duplicate image groups; these are not patient IDs."""
import argparse
import csv
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--images', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--threshold', type=float, default=0.99955)
    args = parser.parse_args()
    paths = sorted(Path(args.images).glob('*.tif'))
    if len(paths) != 1000:
        raise ValueError(f'Expected 1000 TIFFs, got {len(paths)}')
    vectors = []
    for path in paths:
        with Image.open(path) as img:
            pixels = np.array(img.resize((64, 64)), dtype=np.float32).ravel()
        pixels -= pixels.mean()
        norm = np.linalg.norm(pixels)
        if norm == 0:
            raise ValueError(f'Constant image: {path}')
        vectors.append(pixels / norm)
    features = np.stack(vectors)
    n_groups, groups = connected_components(csr_matrix(features @ features.T > args.threshold), directed=False)
    with Path(args.output).open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['filename', 'inferred_slice_group'])
        writer.writerows(zip((p.name for p in paths), groups.tolist()))
    print(f'{len(paths)} images, {n_groups} inferred groups, correlation threshold {args.threshold}')
    print('Image-based groups do not establish patient identity; validate against source metadata.')


if __name__ == '__main__':
    main()
