"""Fixed training-only GradCAM audit; heatmaps are not anatomical annotations."""
import csv,json,random,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
import numpy as np
from PIL import Image,ImageDraw
from inference import Predictor,resolve_images,file_sha256

def main():
    checkpoint=Path('/home/ng/workspace/glaucoma_three22k_crop_native_20261102/outputs/three22k_crop_native_v1/seed42.pt')
    assert file_sha256(checkpoint)=='a2f1ac437eed16b5fc1ec9fdf5c4cee12d469918847680b156f94a4b9f4278ba'
    p=Predictor(checkpoint)
    for parameter in p.model.parameters():parameter.requires_grad_(False)
    rows=list(csv.DictReader((ROOT/'datasets/glaucoma_first/train.csv').open()));rng=random.Random(20261104)
    selected=[]
    for label in ('1','0'):selected+=rng.sample([r for r in rows if r['True_Label']==label],24)
    paths=resolve_images([r['Image_name'] for r in selected],ROOT/'datasets/glaucoma_first/train')
    out=ROOT/'outputs/training_cam_v1';out.mkdir(exist_ok=False)
    sheet=Image.new('RGB',(6*256,8*280),'white');draw=ImageDraw.Draw(sheet);records=[]
    for i,(row,path) in enumerate(zip(selected,paths)):
        tensor=p.preprocess(path).unsqueeze(0).cuda().requires_grad_(True)
        with torch.enable_grad(),torch.autocast('cuda',dtype=p.autocast_dtype):
            features=p.model.forward_features(tensor);features.retain_grad();score=p.model.forward_head(features).float().flatten()[0]
        score.backward();weights=features.grad.float().mean((2,3),keepdim=True)
        cam=(weights*features.float()).sum(1).relu().detach();cam=torch.nn.functional.interpolate(cam.unsqueeze(1),size=(256,256),mode='bilinear',align_corners=False)[0,0].cpu().numpy()
        cam/=max(float(cam.max()),1e-12)
        rgb=tensor.detach()[0].cpu().float().permute(1,2,0).numpy()*np.array([.229,.224,.225])+np.array([.485,.456,.406])
        image=Image.fromarray(np.uint8(np.clip(rgb,0,1)*255)).resize((256,256))
        heat=np.zeros((256,256,3),dtype=np.uint8);heat[:,:,0]=255;heat[:,:,1]=np.uint8(cam*80)
        overlay=np.asarray(image).astype(float)*(.9-.4*cam[:,:,None])+heat*(.1+.4*cam[:,:,None]);preview=Image.fromarray(np.uint8(np.clip(overlay,0,255)))
        x,y=i%6*256,i//6*280;sheet.paste(preview,(x,y));prob=float(score.detach().sigmoid());draw.text((x+2,y+257),f"{row['Image_name']} y={row['True_Label']} p={prob:.3f}",fill='black')
        peak=np.unravel_index(cam.argmax(),cam.shape);records.append(dict(image_name=row['Image_name'],label=int(row['True_Label']),probability=prob,peak_xy=[int(peak[1]),int(peak[0])],cam_nonzero=bool(cam.max()>0)))
        del tensor,features,score;print('CAM',i+1,flush=True)
    sheet.save(out/'training_cam.jpg');(out/'selection.json').write_text(json.dumps(dict(seed=20261104,split='train',checkpoint_sha256=file_sha256(checkpoint),method='Positive-logit GradCAM at last spatial feature map; diagnostic only',rows=records),indent=2)+'\n')

if __name__=='__main__':main()
