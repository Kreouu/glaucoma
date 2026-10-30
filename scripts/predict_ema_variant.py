"""Generate ordered CSV using a frozen native-logit/probability EMA recipe."""
import argparse,csv,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from ema_variant_inference import EMAVariantPredictor
from inference import file_sha256,resolve_images,write_prediction


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--recipe',type=Path,required=True);parser.add_argument('--names-csv',type=Path,required=True)
    parser.add_argument('--images',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    with args.names_csv.open(encoding='utf-8-sig',newline='') as stream:names=[r['Image_name'] for r in csv.DictReader(stream)]
    if not names:parser.error('No input images')
    paths=resolve_images(names,args.images);predictor=EMAVariantPredictor(args.recipe);args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x',newline='') as stream:
        w=csv.writer(stream);w.writerow(['Image_name','Probability','Pred_Label']);stream.flush()
        for step,(name,path) in enumerate(zip(names,paths),1):
            probability,label=predictor.predict_one(path);write_prediction(w,stream,name,probability,label)
            if step%500==0 or step==len(names):print(f'{step}/{len(names)}',flush=True)
    result=dict(rows=len(names),recipe_sha256=file_sha256(args.recipe),weights_manifest_sha256=predictor.recipe['weights_manifest_sha256'],
                names_csv_sha256=file_sha256(args.names_csv),output_sha256=file_sha256(args.output),script_sha256=file_sha256(Path(__file__)),
                aggregation=predictor.recipe['aggregation'],horizontal_flip=predictor.recipe['horizontal_flip'])
    args.output.with_suffix('.manifest.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()
