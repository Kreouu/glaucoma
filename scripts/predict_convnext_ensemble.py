"""Predict in a supplied CSV's image order using a verified equal-probability manifest."""
import argparse
import csv
import json
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from ensemble_inference import ConvNeXtProbabilityPredictor
from inference import file_sha256,resolve_images,write_prediction


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--names-csv',type=Path,required=True)
    parser.add_argument('--images',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    with args.names_csv.open(encoding='utf-8-sig',newline='') as stream:
        names=[r['Image_name'] for r in csv.DictReader(stream)]
    if not names:parser.error('Input CSV has no images')
    paths=resolve_images(names,args.images)
    predictor=ConvNeXtProbabilityPredictor(args.manifest)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x',newline='') as stream:
        writer=csv.writer(stream);writer.writerow(['Image_name','Probability','Pred_Label'])
        for step,(name,path) in enumerate(zip(names,paths),1):
            probability,label=predictor.predict_one(path)
            write_prediction(writer,stream,name,probability,label)
            if step%500==0 or step==len(names):print(f'{step}/{len(names)}',flush=True)
    args.output.with_suffix('.manifest.json').write_text(json.dumps(dict(
        rows=len(names),inference_manifest_sha256=file_sha256(args.manifest),
        names_csv_sha256=file_sha256(args.names_csv),output_sha256=file_sha256(args.output),
        source_sha256=file_sha256(Path(__file__))),indent=2)+'\n')


if __name__=='__main__':main()
