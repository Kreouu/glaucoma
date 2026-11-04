"""Training-only positive-logit CAM flip stability; no fitting or anatomical labels."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from inference import Predictor,resolve_images

def main():
    selection=json.loads((ROOT/'outputs/training_cam_v1/selection.json').read_text());rows=selection['rows']
    p=Predictor('/home/ng/workspace/glaucoma_three22k_crop_native_20261102/outputs/three22k_crop_native_v1/seed42.pt')
    for parameter in p.model.parameters():parameter.requires_grad_(False)
    def cam(tensor):
        tensor=tensor.detach().requires_grad_(True)
        with torch.enable_grad(),torch.autocast('cuda',dtype=p.autocast_dtype):
            feature=p.model.forward_features(tensor);feature.retain_grad();value=p.model.forward_head(feature).float().sum()
        value.backward();weight=feature.grad.float().mean((2,3),keepdim=True)
        result=(weight*feature.float()).sum(1).relu().detach()
        return torch.nn.functional.interpolate(result[:,None],size=(512,512),mode='bilinear',align_corners=False)[0,0].cpu().numpy()
    paths=resolve_images([r['image_name'] for r in rows],ROOT/'datasets/glaucoma_first/train');results=[]
    for row,path in zip(rows,paths):
        x=p.preprocess(path).unsqueeze(0).cuda();a=cam(x);b=cam(x.flip(-1))[:,::-1]
        rgb=x.detach()[0].float().cpu().permute(1,2,0).numpy()*np.array([.229,.224,.225])+np.array([.485,.456,.406]);mask=rgb.max(2)>.08
        mass=float(a[mask].sum()/max(a.sum(),1e-12));valid=bool(a.max()>0 and b.max()>0 and mask.any())
        result=dict(image_name=row['image_name'],label=row['label'],valid=valid,foreground_activation_fraction=mass)
        if valid:
            aa=np.where(mask,a,-1);bb=np.where(mask,b,-1);pa=np.array(np.unravel_index(aa.argmax(),aa.shape));pb=np.array(np.unravel_index(bb.argmax(),bb.shape))
            topa=mask & (a>=np.quantile(a[mask],.9));topb=mask & (b>=np.quantile(b[mask],.9))
            result.update(masked_peak_distance_fraction=float(np.linalg.norm(pa-pb)/512),top10_iou=float((topa&topb).sum()/max(1,(topa|topb).sum())))
        results.append(result)
    summaries={}
    for label in (0,1):
        group=[r for r in results if r['label']==label];valid=[r for r in group if r['valid']]
        summaries[str(label)]=dict(images=len(group),valid=len(valid),median_foreground_mass=float(np.median([r['foreground_activation_fraction'] for r in group])),median_masked_peak_distance=float(np.median([r['masked_peak_distance_fraction'] for r in valid])) if valid else None,median_top10_iou=float(np.median([r['top10_iou'] for r in valid])) if valid else None)
    out=ROOT/'outputs/training_cam_stability_v1';out.mkdir(exist_ok=False)
    report=dict(status='DIAGNOSTIC_ONLY',images=len(results),split='train',anatomical_localization_accuracy_unknown=True,method='Positive-logit GradCAM, inverse horizontal flip, fixedRGBforeground>.08; no fitting',groups=summaries,rows=results)
    (out/'summary.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(summaries),flush=True)

if __name__=='__main__':main()
