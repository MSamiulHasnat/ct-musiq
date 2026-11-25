import unittest

import torch

from ct_musiq.model import CTMUSIQ


class ScaleContractTest(unittest.TestCase):
    def test_single_two_and_three_scale_forward_backward(self):
        for scales in ([16], [16, 24], [16, 24, 32]):
            with self.subTest(scales=scales):
                model = CTMUSIQ(num_scales=len(scales), scales=scales,
                                d_model=32, num_heads=4, num_layers=1,
                                patch_size=8, max_grid_size=2, pretrained=False, dropout=0)
                coords = []
                for scale_index, size in enumerate(scales):
                    grid = size // 8
                    coords.extend([scale_index, row, col]
                                  for row in range(grid) for col in range(grid))
                coordinates = torch.tensor(coords).unsqueeze(0).repeat(2, 1, 1)
                patches = torch.randn(2, len(coords), 3, 8, 8)
                output = model(patches, coordinates)
                self.assertEqual(output['score'].shape, (2, 1))
                self.assertEqual(len(output['scale_scores']), len(scales))
                loss = output['score'].sum() + sum(score.sum() for score in output['scale_scores'])
                self.assertTrue(torch.isfinite(loss))
                loss.backward()
                for head in model.scale_heads:
                    self.assertIsNotNone(head[0].weight.grad)

    def test_mismatched_scale_count_rejected(self):
        with self.assertRaises(ValueError):
            CTMUSIQ(num_scales=2, scales=[16], pretrained=False)


if __name__ == '__main__':
    unittest.main()
