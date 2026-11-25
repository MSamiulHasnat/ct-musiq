"""Exploratory evaluation of an existing checkpoint on the released test set.

Run from a copy of the repository, with --data pointing to the directory
containing test.json and images/. This does not train or select a model.
"""
import argparse
import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import torch
import scipy
import timm
from torch.utils.data import DataLoader, Dataset

import config
from dataset import LDCTDataset
from evaluate import compute_metrics, evaluate_model
from model import create_model


class ReleasedTest(Dataset):
    def __init__(self, root, scales):
        self.root = Path(root)
        self.labels = json.loads((self.root / 'test.json').read_text())
        self.names = sorted(self.labels)
        if len(self.names) != 300:
            raise ValueError(f'Expected 300 labels, found {len(self.names)}')
        self.transform = LDCTDataset.__new__(LDCTDataset)
        self.transform.scales = scales
        self.transform.patch_size = config.PATCH_SIZE

    def __len__(self):
        return len(self.names)

    def __getitem__(self, index):
        name = self.names[index]
        img = np.clip(np.asarray(Image.open(self.root / 'images' / name), dtype=np.float32), 0, 1)
        patches, coordinates = [], []
        for scale, image in enumerate(self.transform.build_multi_scale_pyramid(img)):
            p, c = self.transform.extract_patches(image, scale)
            patches.append(p)
            coordinates.append(c)
        patches = self.transform.replicate_to_rgb(np.concatenate(patches))
        return {'patches': torch.from_numpy(patches).float(),
                'coords': torch.from_numpy(np.concatenate(coordinates)).long(),
                'score': torch.tensor(self.labels[name], dtype=torch.float32),
                'image_id': name}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--batch-size', type=int, default=4)
    args = parser.parse_args()
    checkpoint_path = Path(args.checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    scales = checkpoint.get('config', {}).get('scales', config.SCALES)
    if scales != config.SCALES:
        raise ValueError(f'Checkpoint scales {scales} do not match configuration {config.SCALES}')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = create_model(num_scales=len(scales), pretrained=False, device=device)
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    loader = DataLoader(ReleasedTest(args.data, scales), batch_size=args.batch_size, shuffle=False)
    predictions, targets, names = evaluate_model(model, loader, device, model_type='ct_musiq')
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({'image_id': names, 'predicted': predictions, 'target': targets}).to_csv(output / 'predictions.csv', index=False)
    metrics = {key: float(value) for key, value in compute_metrics(predictions, targets).items()}
    metrics.update({'n': len(targets), 'mae': float(np.mean(np.abs(predictions-targets))),
                    'checkpoint_sha256': hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
                    'checkpoint_epoch_zero_based': checkpoint.get('epoch'),
                    'checkpoint_config': checkpoint.get('config'), 'torch': torch.__version__,
                    'scipy': scipy.__version__, 'timm': timm.__version__,
                    'python': platform.python_version(), 'cuda_runtime': torch.version.cuda,
                    'gpu': torch.cuda.get_device_name(device) if device.type == 'cuda' else None,
                    'device': str(device), 'tta': False})
    (output / 'metrics.json').write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == '__main__':
    main()
