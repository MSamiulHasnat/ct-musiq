"""A compact high-resolution transfer baseline; no novelty claim is implied."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
import timm
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from safetensors.torch import load_file

from ct_musiq.evaluate import compute_metrics
from ct_musiq.loss import CTMUSIQLoss
from ct_musiq.train import set_seed


class Images(Dataset):
    def __init__(self, directory, labels, names, resolution, augment=False, quantize_uint8=False):
        self.directory, self.labels, self.names = Path(directory), labels, names
        self.resolution, self.augment = resolution, augment
        self.quantize_uint8 = quantize_uint8

    def __len__(self):
        return len(self.names)

    def __getitem__(self, index):
        name = self.names[index]
        with Image.open(self.directory / name) as image:
            pixels = np.clip(np.array(image, dtype=np.float32), 0, 1)
        if self.quantize_uint8:
            pixels = (pixels * 255).astype(np.uint8).astype(np.float32) / 255
        if self.augment:
            if np.random.random() < 0.5:
                pixels = pixels[:, ::-1].copy()
            if np.random.random() < 0.5:
                pixels = pixels[::-1, :].copy()
        image = Image.fromarray(pixels, mode='F').resize((self.resolution, self.resolution), Image.Resampling.BICUBIC)
        pixels = np.clip(np.array(image, dtype=np.float32), 0, 1)
        return torch.from_numpy(pixels).unsqueeze(0).repeat(3, 1, 1), float(self.labels[name]), name


class CompactIQA(nn.Module):
    def __init__(self, pretrained_file=None, backbone='efficientnet_b0'):
        super().__init__()
        self.backbone = timm.create_model(backbone, pretrained=False)
        if pretrained_file:
            self.backbone.load_state_dict(load_file(pretrained_file), strict=True)
        width = self.backbone.num_features
        self.backbone.reset_classifier(0)
        self.head = nn.Sequential(nn.Linear(width, 256), nn.GELU(), nn.Dropout(0.1), nn.Linear(256, 1))
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, images):
        return 4 * torch.sigmoid(self.head(self.backbone((images-self.mean)/self.std)))


@torch.inference_mode()
def predict(model, loader, device):
    model.eval()
    names, predictions, targets = [], [], []
    for images, scores, ids in loader:
        predictions.extend(model(images.to(device)).flatten().cpu().tolist())
        targets.extend(scores.tolist())
        names.extend(ids)
    return names, np.array(predictions), np.array(targets)


def save_predictions(path, names, pred, target):
    with path.open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['image_id', 'predicted', 'target'])
        writer.writerows(zip(names, pred, target))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image-dir', required=True)
    parser.add_argument('--labels', required=True)
    parser.add_argument('--split')
    parser.add_argument('--pretrained-file')
    parser.add_argument('--checkpoint', help='Evaluate an existing compact checkpoint instead of training')
    parser.add_argument('--output', required=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--resolution', type=int, default=448)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--quantize-uint8', action='store_true',
                        help='Controlled ablation: quantize to 8-bit before the unchanged float resize')
    args = parser.parse_args()
    set_seed(args.seed)
    labels = json.loads(Path(args.labels).read_text())
    out = Path(args.output)
    if not args.checkpoint and out.exists():
        raise FileExistsError(f'Choose a new output directory: {out}')
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=True) if args.checkpoint else None
    resolution = checkpoint['config']['resolution'] if checkpoint else args.resolution
    quantize = checkpoint['config'].get('quantize_uint8', False) if checkpoint else args.quantize_uint8
    model = CompactIQA(None if checkpoint else args.pretrained_file).to(device)
    if checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'], strict=True)
        loader = DataLoader(Images(args.image_dir, labels, sorted(labels), resolution,
                                   quantize_uint8=quantize), batch_size=args.batch_size, num_workers=args.workers)
        names, pred, target = predict(model, loader, device)
        metrics = {key: float(value) for key, value in compute_metrics(pred, target).items()}
        metrics.update(n=len(target), mae=float(np.abs(pred-target).mean()), checkpoint_config=checkpoint['config'])
        save_predictions(out/'predictions.csv', names, pred, target)
        (out/'metrics.json').write_text(json.dumps(metrics, indent=2))
        print(json.dumps(metrics, indent=2), flush=True)
        return
    if not args.split or not args.pretrained_file:
        parser.error('Training requires --split and --pretrained-file')
    split = json.loads(Path(args.split).read_text())
    if set(split['train']) & set(split['val']) or set(split['train']+split['val']) != set(labels):
        raise ValueError('Development split must cover labels without overlap')
    train_loader = DataLoader(Images(args.image_dir, labels, split['train'], resolution, True, quantize), batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)
    val_loader = DataLoader(Images(args.image_dir, labels, split['val'], resolution,
                                   quantize_uint8=quantize), batch_size=args.batch_size, num_workers=args.workers, pin_memory=True)
    criterion = CTMUSIQLoss(lambda_kl=0, lambda_rank=0.5).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=1e-6)
    scaler = torch.amp.GradScaler('cuda', enabled=device.type == 'cuda')
    best, patience = float('-inf'), 0
    config = {'model': 'efficientnet_b0', 'resolution': resolution, 'seed': args.seed,
              'batch_size': args.batch_size, 'epochs': args.epochs, 'lr': 1e-4, 'lambda_rank': 0.5,
              'quantize_uint8': quantize,
              'split_sha256': hashlib.sha256(Path(args.split).read_bytes()).hexdigest(),
              'pretrained_sha256': hashlib.sha256(Path(args.pretrained_file).read_bytes()).hexdigest()}
    with (out/'training_log.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['epoch', 'train_loss', 'PLCC', 'SROCC', 'KROCC', 'Aggregate', 'mae'])
        writer.writeheader()
        for epoch in range(args.epochs):
            model.train()
            total, count = 0.0, 0
            for images, scores, _ in train_loader:
                images, scores = images.to(device), scores.float().to(device)
                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast('cuda', enabled=device.type == 'cuda'):
                    loss = criterion({'score': model(images), 'scale_scores': []}, scores)['total']
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                total += loss.item() * len(scores)
                count += len(scores)
            scheduler.step()
            names, pred, target = predict(model, val_loader, device)
            metrics = compute_metrics(pred, target)
            row = {'epoch': epoch+1, 'train_loss': total/count,
                   **{key: float(metrics[key]) for key in ['PLCC', 'SROCC', 'KROCC', 'Aggregate']},
                   'mae': float(np.abs(pred-target).mean())}
            writer.writerow(row)
            stream.flush()
            score = row['PLCC'] + row['SROCC'] + row['KROCC']
            print(json.dumps(row), flush=True)
            if score > best:
                best, patience = score, 0
                torch.save({'model_state_dict': model.state_dict(), 'config': config, 'epoch': epoch,
                            'validation': row}, out/'best.pth')
                save_predictions(out/'validation_predictions.csv', names, pred, target)
            else:
                patience += 1
            if epoch >= 19 and patience >= 20:
                break


if __name__ == '__main__':
    main()
