"""Compare frozen models' FP32 GPU forwards on the same development images.

Excludes CPU preprocessing, disk I/O, host-device transfer and model loading.
This is a single-machine timing, not a clinical deployment measurement.
"""
import argparse
import json
from pathlib import Path
import time

import torch
from ct_musiq.model import create_model
from ct_musiq.dataset import LDCTDataset
from research_audit.compact_iqa import CompactIQA, Images


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--compact-checkpoint', required=True)
    parser.add_argument('--aligned-checkpoint', required=True)
    parser.add_argument('--images', required=True)
    parser.add_argument('--labels', required=True)
    parser.add_argument('--split', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('This benchmark requires CUDA')
    device = torch.device('cuda')
    labels = json.loads(Path(args.labels).read_text())
    names = sorted(json.loads(Path(args.split).read_text())['train'])[:32]
    compact_checkpoint = torch.load(args.compact_checkpoint, map_location='cpu', weights_only=True)
    aligned_checkpoint = torch.load(args.aligned_checkpoint, map_location='cpu', weights_only=True)
    result = {'torch': str(torch.__version__), 'gpu': torch.cuda.get_device_name(),
              'precision': 'float32, no autocast', 'tf32_matmul': torch.backends.cuda.matmul.allow_tf32,
              'tf32_cudnn': torch.backends.cudnn.allow_tf32,
              'warmup_forwards': 10, 'measured_forwards_per_repeat': 20, 'repeats': 5,
              'excluded': ['CPU preprocessing', 'disk I/O', 'host-device transfer', 'model loading'],
              'input_ids': names, 'models': {}}
    for kind in ['compact', 'aligned']:
        if kind == 'compact':
            model = CompactIQA()
            checkpoint = compact_checkpoint
            data = Images(args.images, labels, names, checkpoint['config']['resolution'])
            inputs = [torch.stack([data[i][0] for i in range(32)]).to(device)]
        else:
            checkpoint = aligned_checkpoint
            settings = checkpoint['config']
            model = create_model(num_scales=len(settings['scales']), scales=settings['scales'],
                                 profile=settings['profile'], pretrained=False, device='cpu')
            data = LDCTDataset(data_dir=args.images, label_file=args.labels, split='train',
                               scales=settings['scales'], image_ids=names, augment=False, float_resize=True)
            rows = [data[i] for i in range(32)]
            inputs = [torch.stack([row[key] for row in rows]).to(device) for key in ['patches', 'coords']]
        model.load_state_dict(checkpoint['model_state_dict'], strict=True)
        model = model.to(device).eval()
        record = {'parameters': sum(p.numel() for p in model.parameters()),
                  'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
                  'batches': {}}
        with torch.inference_mode():
            for batch in [1, 32]:
                values = [value[:batch] for value in inputs]
                for _ in range(10):
                    model(*values)
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                times = []
                for _ in range(5):
                    start = time.perf_counter()
                    for _ in range(20):
                        model(*values)
                    torch.cuda.synchronize()
                    times.append((time.perf_counter()-start)/20)
                middle = sorted(times)[2]
                record['batches'][str(batch)] = {
                    'repeat_seconds_per_forward': times, 'median_seconds_per_forward': middle,
                    'images_per_second': batch/middle,
                    'peak_allocated_bytes': torch.cuda.max_memory_allocated()}
        result['models'][kind] = record
        del model, inputs, values
        torch.cuda.empty_cache()
    Path(args.output).write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
