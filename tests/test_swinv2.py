"""Resolution-transfer regression: constructor must work at 384 and preserve weights."""

import importlib.util
import unittest
from pathlib import Path


@unittest.skipUnless(importlib.util.find_spec('timm'), 'optional timm dependency not installed')
class SwinResolutionTests(unittest.TestCase):
    def test_384_window_geometry(self):
        from model_swinv2 import build_model
        model = build_model(size=384)
        self.assertEqual(model.patch_embed.img_size, (384, 384))
        self.assertEqual(model.num_classes, 1)
        for stage in model.layers:
            for block in stage.blocks:
                self.assertEqual(block.window_size, (12, 12))
                self.assertTrue(all(r % 12 == 0 for r in block.input_resolution))

    def test_pretrained_learned_tensors_preserved(self):
        import torch
        from model_swinv2 import build_model
        path = Path('pretrain_model/swinv2_tiny_window8_256_ms_in1k.bin')
        if not path.exists():
            self.skipTest('original weights not available')
        original = torch.load(path, map_location='cpu', weights_only=True)
        model = build_model(path, size=384)
        observed = model.state_dict()
        for name, value in original.items():
            if name.startswith('head.') or name.endswith('attn_mask'):
                continue
            self.assertIn(name, observed)
            self.assertTrue(torch.equal(value, observed[name]), name)


if __name__ == '__main__':
    unittest.main()
