"""Fixed three-seed22K cosine plus training crop comparison."""
import hashlib,json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def main():
 plan=json.loads((ROOT/'docs/evidence/convnext22k_crop640_recovery_v1/plan.json').read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
 assert json.loads(Path(plan['terminal_proof']).read_text())['status']=='PASS'
 for n,h in plan['source_sha256'].items():assert sha(ROOT/n)==h,n
 for split,h in plan['csv_sha256'].items():assert sha(ROOT/f'datasets/glaucoma_first/{split}.csv')==h
 assert sha(ROOT/plan['pretrained_path'])==plan['pretrained_sha256']
 out=ROOT/'outputs/convnext22k_crop640_recovery_v1';out.mkdir(exist_ok=False);(out/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
 def run(seed,smoke=False):
  name='smoke' if smoke else f'seed{seed}';p=out/name
  cmd=[sys.executable,'-u','train.py','--model','convnext_tiny','--precision','fp16','--data-root','datasets/glaucoma_first','--pretrained-path',plan['pretrained_path'],'--batch-size','16','--size','640','--jpeg-draft','--workers','4','--lr','.0001','--lr-schedule','warmup_cosine','--warmup-epochs','1','--min-lr','.00001','--train-crop-min-area','.8','--pos-weight','3','--ema-decay','.999','--epochs','2' if smoke else '5','--seed',str(seed),'--output-dir',str(p)]
  if smoke:cmd+=['--train-limit','64','--valid-limit','64']
  print('START',name,flush=True)
  with (out/(name+'.log')).open('x') as f:subprocess.run(cmd,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True)
  for kind in ('','ema'):
   part=p/kind;subprocess.run([sys.executable,'validate.py','--run-dir',str(part)],cwd=ROOT,check=True)
   c=json.loads((part/'reload_check.json').read_text());assert c['status']=='PASS' and c['images']==(64 if smoke else 5849) and c['max_probability_difference']==0 and sha(part/'best.pt')==c['checkpoint_sha256']
  config=json.loads((p/'ema/config.json').read_text());assert config['train_crop_min_area']==.8 and config['lr_schedule']=='warmup_cosine' and config['pretrained_sha256']==plan['pretrained_sha256']
  entry=dict(seed=seed,check=c);print('DONE',json.dumps(entry),flush=True);return entry
 run(42,True);results=[run(42)]
 (out/'progress.json').write_text(json.dumps(results,indent=2)+'\n')
 if results[0]['check']['val_auroc']>=plan['confirmation_auroc']:
  for seed in (43,44):
   results.append(run(seed));(out/'progress.json').write_text(json.dumps(results,indent=2)+'\n')
 (out/'summary.json').write_text(json.dumps(dict(status='PASS',results=results,test_used=False,scope='640 resolution vs512 same22K cosine crop recipe;42gate then43/44; no test use'),indent=2)+'\n')
if __name__=='__main__':main()
