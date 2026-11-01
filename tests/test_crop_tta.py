import random,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import torch
from PIL import Image
from torchvision import transforms
from dataset import Letterbox
from crop_tta import preprocess_center,center_crop
from training_crop import RandomRelativeCrop


class CropTTATest(unittest.TestCase):
    def test_matches_forced_training_crop_after_jpeg_draft(self):
        transform=transforms.Compose([Letterbox(512),transforms.ToTensor(),transforms.Normalize([.485,.456,.406],[.229,.224,.225])])
        predictor=SimpleNamespace(config={'jpeg_draft':True,'size':512},transform=transform)
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'probe.jpg';Image.linear_gradient('L').resize((1920,1080)).convert('RGB').save(path)
            state=random.getstate();actual=preprocess_center(path,predictor);self.assertEqual(random.getstate(),state)
            with Image.open(path) as image:
                image.draft('RGB',(512,512));image=image.convert('RGB');cropped=center_crop(image);left=(image.width-cropped.width)//2;top=(image.height-cropped.height)//2
                with patch('training_crop.random.uniform',return_value=.9),patch('training_crop.random.randint',side_effect=[left,top]):expected=transform(RandomRelativeCrop(.8)(image))
            self.assertTrue(torch.equal(actual,expected));self.assertEqual(tuple(actual.shape),(3,512,512));self.assertTrue(torch.isfinite(actual).all())


if __name__=='__main__':unittest.main()
