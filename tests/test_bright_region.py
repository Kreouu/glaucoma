import random,sys,tempfile,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from PIL import Image,ImageDraw
from torchvision import transforms
from dataset import Letterbox
from fundus_bright_crop import bright_box,BrightRegionDataset
from inference import Predictor


class BrightRegionTest(unittest.TestCase):
    def test_fixed_localization_bounds_and_fallback(self):
        image=Image.new('RGB',(512,512));draw=ImageDraw.Draw(image);draw.ellipse((30,30,482,482),fill=(70,30,15));draw.ellipse((350,230,400,280),fill=(240,210,90))
        state=random.getstate();box=bright_box(image);self.assertEqual(random.getstate(),state)
        self.assertEqual(box,bright_box(image));x0,y0,x1,y1=box
        self.assertEqual(x1-x0,y1-y0);self.assertTrue(0<=x0<x1<=512 and 0<=y0<y1<=512);self.assertTrue(x0<=375<=x1 and y0<=255<=y1)
        self.assertIsNone(bright_box(Image.new('RGB',(512,512))))
        glare=Image.new('RGB',(512,512),(70,30,15));ImageDraw.Draw(glare).ellipse((200,200,300,300),fill=(250,250,250));self.assertIsNone(bright_box(glare))

    def test_dataset_and_inference_jpeg_path_exact(self):
        transform=transforms.Compose([Letterbox(512),transforms.ToTensor(),transforms.Normalize([.485,.456,.406],[.229,.224,.225])])
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'probe.jpg';image=Image.new('RGB',(2560,1920),(70,30,15));ImageDraw.Draw(image).ellipse((1500,700,1750,950),fill=(240,210,90));image.save(path)
            dataset=object.__new__(BrightRegionDataset);dataset.samples=[('probe.jpg',path,1)];dataset.size=512;dataset.jpeg_draft=True;dataset.transform=transform
            predictor=object.__new__(Predictor);predictor.config=dict(bright_region_crop=True,size=512,jpeg_draft=True);predictor.transform=transform
            a,label,name=dataset[0];b=predictor.preprocess(path)
            self.assertTrue(torch.equal(a,b));self.assertEqual(tuple(a.shape),(3,512,512));self.assertTrue(torch.isfinite(a).all());self.assertEqual(label.item(),1.)


if __name__=='__main__':unittest.main()
