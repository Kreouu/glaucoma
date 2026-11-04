"""Frozen label-independent CAM regions for official train/valid; never test."""
import csv,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from PIL import Image,ImageOps
from inference import Predictor,resolve_images,file_sha256
from model_convnext import build_model
from dataset import Letterbox
from torchvision import transforms
from sklearn.metrics import roc_auc_score,average_precision_score
CHECKPOINT=Path('/home/ng/workspace/glaucoma_three22k_crop_native_20261102/outputs/three22k_crop_native_v1/seed42.pt')

def main():
    assert file_sha256(CHECKPOINT)=='a2f1ac437eed16b5fc1ec9fdf5c4cee12d469918847680b156f94a4b9f4278ba'
    p=Predictor(CHECKPOINT)
    for v in p.model.parameters():v.requires_grad_(False)
    out=ROOT/'outputs/cam_local_fast_four_native_v1';out.mkdir(exist_ok=False)
    source=Path('/home/ng/workspace/glaucoma_cam_local_20261104/outputs/cam_local_v1/lr0.0001_seed42/ema/best.pt')
    assert file_sha256(source)=='ad0daddde1f1b5f34d07d8dabdfaa110451153910415bd5b2dc92c6b837393a4'
    saved=torch.load(source,map_location='cpu',weights_only=True);local=build_model();local.load_state_dict(saved['model'],strict=True);local.eval().cuda()
    transform=transforms.Compose([Letterbox(512),transforms.ToTensor(),transforms.Normalize([.485,.456,.406],[.229,.224,.225])])
    cached=json.loads((ROOT/'outputs/cam_region_cache_v1/valid.json').read_text())['regions']
    probabilities=[];labels=[];box_difference=0.
    globals=[Predictor(ROOT/f'outputs/extra_seeds_native_v1/seed{seed}.pt') for seed in (43,44)]
    for split in ('valid',):
        rows=list(csv.DictReader((ROOT/f'datasets/glaucoma_first/{split}.csv').open()));names=[r['Image_name'] for r in rows]
        paths=resolve_images(names,ROOT/f'datasets/glaucoma_first/{split}');regions={};fallbacks=0
        for i,(name,path) in enumerate(zip(names,paths),1):
            x=p.preprocess(path).unsqueeze(0).cuda()
            with torch.enable_grad(),torch.autocast('cuda',dtype=p.autocast_dtype):
                with torch.no_grad():f=p.model.forward_features(x)
                f=f.detach().requires_grad_(True);score=p.model.forward_head(f).float().sum()
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
            box_difference=max(box_difference,max(abs(a-b) for a,b in zip(box,cached[name]['box_fraction_xyxy'])))
            assert box_difference==0
            with Image.open(path) as image:
                if image.format=='JPEG':image.draft('RGB',(512,512))
                image=image.convert('RGB');iw,ih=image.size
                cropbox=(int(box[0]*iw),int(box[1]*ih),max(int(box[0]*iw)+1,round(box[2]*iw)),max(int(box[1]*ih)+1,round(box[3]*ih)))
                tensor=transform(image.crop(cropbox)).unsqueeze(0).cuda()
            with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16):
                logits=[score.detach()]+[g.model(x).float().flatten()[0] for g in globals]+[local(tensor).float().flatten()[0]]
                probability=torch.stack(logits).float().mean().sigmoid().item()
            probabilities.append(probability);labels.append(int(rows[i-1]['True_Label']))
            if i%1000==0:print(split,i,flush=True)
            del x,f,score
        with (out/'valid_predictions.csv').open('x') as stream:
            writer=csv.writer(stream);writer.writerow(['Image_name','True_Label','Probability']);writer.writerows(zip(names,labels,probabilities))
        reference=list(csv.DictReader((ROOT/'outputs/cam_local_four_native_v1/valid_predictions.csv').open()))
        assert [r['Image_name'] for r in reference]==names
        difference=max(abs(a-float(b['Probability'])) for a,b in zip(probabilities,reference));assert difference<=1e-6
        report=dict(max_probability_difference_vs_full_backprop=difference,status='DYNAMIC_NATIVE_VALIDATION',images=len(probabilities),max_box_difference=box_difference,auroc=float(roc_auc_score(labels,probabilities)),ap=float(average_precision_score(labels,probabilities)),checkpoint_sha256=file_sha256(source),test_used=False,runtime_verified=False)
        (out/'verification.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)

if __name__=='__main__':main()
