import unittest

import numpy as np
import torch

from ct_musiq.dataset import LDCTDataset
from ct_musiq.model import CTMUSIQ


class TransferProfileTest(unittest.TestCase):
    def test_float_resize_preserves_sub_byte_precision(self):
        data = LDCTDataset.__new__(LDCTDataset)
        data.scales = [32]
        data.float_resize = True
        image = np.full((32, 32), 0.123456, dtype=np.float32)
        np.testing.assert_allclose(data.build_multi_scale_pyramid(image)[0], image, atol=1e-7)
        data.float_resize = False
        self.assertGreater(float(np.abs(data.build_multi_scale_pyramid(image)[0] - image).mean()), 1e-4)

    def test_safe_augmentation_preserves_pixel_values(self):
        data = LDCTDataset.__new__(LDCTDataset)
        data.quality_safe_augmentation = True
        image = np.arange(16, dtype=np.float32).reshape(4, 4) / 16
        for _ in range(10):
            np.testing.assert_array_equal(np.sort(data.apply_augmentation(image).ravel()), np.sort(image.ravel()))

    def test_aligned_profile_forward(self):
        model = CTMUSIQ(num_scales=1, scales=[16], d_model=36, num_heads=12,
                        num_layers=1, patch_size=8, pretrained=False,
                        transfer_aligned=True, dropout=0)
        output = model(torch.rand(2, 4, 3, 8, 8), torch.zeros(2, 4, 3, dtype=torch.long))
        self.assertTrue(torch.isfinite(output['score']).all())


if __name__ == '__main__':
    unittest.main()
