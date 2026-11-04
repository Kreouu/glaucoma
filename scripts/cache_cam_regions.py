"""Frozen label-independent CAM regions for official train/valid; never test."""
import csv,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from PIL import Image,ImageOps
from inference import Predictor,resolve_images,file_sha256
CHECKPOINT=Path('/home/ng/workspace/glaucoma_three22k_crop_native_20261102/outputs/three22k_crop_native_v1/seed42.pt')

def main():
    assert file_sha256(CHECKPOINT)=='a2f1ac437eed16b5fc1ec9fdf5c4cee12d469918847680b156f94a4b9f4278ba'
    p=Predictor(CHECKPOINT)
    for v in p.model.parameters():v.requires_grad_(False)
    out=ROOT/'outputs/cam_region_cache_v1';out.mkdir(exist_ok=False)
    for split in ('train','valid'):
        rows=list(csv.DictReader((ROOT/f'datasets/glaucoma_first/{split}.csv').open()));names=[r['Image_name'] for r in rows]
        paths=resolve_images(names,ROOT/f'datasets/glaucoma_first/{split}');regions={};fallbacks=0
        for i,(name,path) in enumerate(zip(names,paths),1):
            x=p.preprocess(path).unsqueeze(0).cuda().requires_grad_(True)
            with torch.enable_grad(),torch.autocast('cuda',dtype=p.autocast_dtype):
                f=p.model.forward_features(x);f.retain_grad();score=p.model.forward_head(f).float().sum()
            score.backward();cam=(f.grad.float().mean((2,3),keepdim=True)*f.float()).sum(1).relu().detach()
            cam=torch.nn.functional.interpolate(cam[:,None],size=(256,256),mode='bilinear',align_corners=False)[0,0]
            rgb=x.detach()[0].float().permute(1,2,0)*x.new_tensor([.229,.224,.225])+x.new_tensor([.485,.456,.406])
            mask=torch.nn.functional.interpolate((rgb.max(2).values>.08).float()[None,None],size=(256,256),mode='nearest')[0,0]
            coverage=torch.nn.functional.avg_pool2d(mask[None,None],154,stride=1)[0,0]
            scores=torch.nn.functional.avg_pool2d((cam*mask)[None,None],154,stride=1)[0,0];allowed=coverage>=.8
            scores=torch.where(allowed,scores,torch.full_like(scores,-1));fallback=not bool(allowed.any()) or float(scores.max())<=0
            if fallback:scores=coverage;fallbacks+=1
            top,left=np.unravel_index(int(scores.argmax()),scores.shape)
            with Image.open(path) as image:
                w,h=image.size
                if p.config['jpeg_draft'] and image.format=='JPEG':image.draft('RGB',(512,512))
                rw,rh=ImageOps.contain(image,(512,512),Image.Resampling.BILINEAR).size
            ox,oy=(512-rw)//2,(512-rh)//2
            box=[max(0,min(1,(2*left-ox)/rw)),max(0,min(1,(2*top-oy)/rh)),max(0,min(1,(2*(left+154)-ox)/rw)),max(0,min(1,(2*(top+154)-oy)/rh))]
            assert 0<=box[0]<box[2]<=1 and 0<=box[1]<box[3]<=1
            regions[name]=dict(box_fraction_xyxy=box,original_size=[w,h],fallback=fallback)
            if i%1000==0:print(split,i,flush=True)
            del x,f,score
        report=dict(split=split,images=len(regions),fallbacks=fallbacks,checkpoint_sha256=file_sha256(CHECKPOINT),csv_sha256=file_sha256(ROOT/f'datasets/glaucoma_first/{split}.csv'),labels_used_for_regions=False,regions=regions)
        (out/f'{split}.json').write_text(json.dumps(report)+'\n');print('DONE',split,len(regions),fallbacks,flush=True)

if __name__=='__main__':main()
