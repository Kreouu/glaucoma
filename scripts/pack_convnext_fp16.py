"""Inference-only storage packing: Conv/Linear weights half, all other tensors unchanged."""
import argparse,copy,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from model_convnext import build_model
from inference import file_sha256


def pack(source,target):
    assert not target.exists()
    saved=torch.load(source,map_location='cpu',weights_only=True)
    assert 'optimizer' not in saved and saved['config'].get('weight_type')=='ema'
    assert saved['config']['model']=='convnext_tiny' and saved['config'].get('precision')=='fp16'
    model=build_model();model.load_state_dict(saved['model'],strict=True)
    converted=set()
    for prefix,module in model.named_modules():
        if isinstance(module,(torch.nn.Conv2d,torch.nn.Linear)):
            for name,value in module.named_parameters(recurse=False):
                if value is not None:converted.add(f'{prefix}.{name}' if prefix else name)
    original=saved['model'];packed={}
    for name,value in original.items():
        assert torch.isfinite(value).all()
        packed[name]=value.half() if name in converted else value.clone()
        assert torch.isfinite(packed[name]).all()
        if name not in converted:assert torch.equal(value,packed[name])
    model.load_state_dict(packed,strict=True)
    for name,value in model.state_dict().items():
        expected=original[name].half().float() if name in converted else original[name]
        assert torch.equal(value,expected),name
    result=copy.copy(saved);result['model']=packed
    result['storage']=dict(inference_only=True,policy='Only Conv2d/Linear parameters FP16; norms, layer scales and other tensors original dtype',source_sha256=file_sha256(source),exporter_sha256=file_sha256(Path(__file__)),converted_keys=sorted(converted))
    target.parent.mkdir(parents=True,exist_ok=True);torch.save(result,target)
    return dict(original_bytes=source.stat().st_size,packed_bytes=target.stat().st_size,original_sha256=file_sha256(source),packed_sha256=file_sha256(target),converted_tensors=len(converted),preserved_tensors=len(original)-len(converted),cpu_tensor_policy_exact=True,gpu_equivalence='NOT_YET_VERIFIED')


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--manifest',type=Path,required=True);parser.add_argument('--output-dir',type=Path,required=True);args=parser.parse_args()
    original=json.loads(args.manifest.read_text());args.output_dir.mkdir(exist_ok=False)
    manifest=copy.deepcopy(original);reports=[]
    for entry in manifest['members']:
        source=args.manifest.parent/entry['file'];assert file_sha256(source)==entry['sha256']
        target=args.output_dir/entry['file'];report=pack(source,target);report['id']=entry['id'];reports.append(report)
        entry.update(sha256=report['packed_sha256'],bytes=report['packed_bytes'],original_sha256=report['original_sha256'])
    manifest['storage_policy']='Selective FP16 storage, restore into FP32 model before the existing FP16 autocast; inference only'
    (args.output_dir/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    result=dict(status='CPU_POLICY_PASS',members=reports,total_original_bytes=sum(r['original_bytes'] for r in reports),total_packed_bytes=sum(r['packed_bytes'] for r in reports),gpu_verified=False)
    (args.output_dir/'packing_report.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)


if __name__=='__main__':main()
