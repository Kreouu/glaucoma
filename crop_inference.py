"""Explicit inference entry for trim-black checkpoints; frozen old loaders stay unchanged."""
from pathlib import Path
from inference import Predictor,file_sha256
from fundus_crop import TrimBlackBorder


class CropPredictor(Predictor):
    def __init__(self,checkpoint):
        super().__init__(checkpoint)
        assert self.config.get('trim_black') is True
        assert file_sha256(Path(__file__).with_name('fundus_crop.py'))==self.config['source_sha256']['fundus_crop.py']
        self.transform.transforms.insert(0,TrimBlackBorder())
