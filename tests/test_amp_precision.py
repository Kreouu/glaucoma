import unittest

import torch

from amp_precision import amp_dtype, precision_description


class PrecisionTests(unittest.TestCase):
    def test_historical_checkpoint_defaults_to_fp16(self):
        self.assertEqual(amp_dtype({'amp': True}), torch.float16)
        self.assertEqual(precision_description({'amp': True}), 'FP16 autocast, FP32 sigmoid')

    def test_bf16_metadata(self):
        self.assertEqual(amp_dtype({'precision': 'bf16'}), torch.bfloat16)
        self.assertEqual(precision_description({'precision': 'bf16'}), 'BF16 autocast, FP32 sigmoid')

    def test_unknown_precision_does_not_silently_fall_back(self):
        with self.assertRaises(ValueError):
            amp_dtype({'precision': 'unknown'})


if __name__ == '__main__':
    unittest.main()
