"""Plot exploratory predictions and score-dependent errors."""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('predictions')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    with open(args.predictions, newline='', encoding='utf-8') as stream:
        rows = list(csv.DictReader(stream))
    target = np.array([float(row['target']) for row in rows])
    pred = np.array([float(row['predicted']) for row in rows])
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.4), constrained_layout=True)
    ax = axes[0]
    ax.scatter(target, pred, s=19, alpha=0.55, color='#276c80', edgecolors='none')
    ax.plot([0, 4], [0, 4], '--', color='#67727a', linewidth=1)
    ax.set(xlabel='Radiologist mean score', ylabel='Predicted score', xlim=(-0.1, 4.1), ylim=(-0.1, 4.1), title='Predicted versus reference')
    ax.set_aspect('equal', adjustable='box')
    ax = axes[1]
    ranges = [(0, 1), (1, 2), (2, 3), (3, 4.001)]
    grouped = [(pred-target)[(target >= low) & (target < high)] for low, high in ranges]
    ax.boxplot(grouped, showfliers=False)
    ax.set_xticks(range(1, 5), ['0–<1', '1–<2', '2–<3', '3–4'])
    ax.axhline(0, linestyle='--', color='#67727a', linewidth=1)
    ax.set(xlabel='Reference score range', ylabel='Prediction − reference', title='Error by reference range')
    fig.suptitle('CT-MUSIQ archived checkpoint · released test set (n=300)', fontsize=12)
    fig.text(0.5, -0.02, 'Exploratory: test images include related slices; groups are not independent patients.', ha='center', fontsize=9, color='#555555')
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200, bbox_inches='tight')


if __name__ == '__main__':
    main()
