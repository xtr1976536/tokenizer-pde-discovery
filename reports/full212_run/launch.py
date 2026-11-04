"""Local process supervisor: single authorized212 run, no automatic large run/retry."""
import hashlib,json,os,subprocess,sys,traceback
from pathlib import Path
from datetime import datetime,timezone
root=Path(__file__).resolve().parents[2];os.chdir(root)
report=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
mlp={str(p):sha(p) for p in sorted((root/'data/mlp_cache').glob('*/surrogate_cache.pt'))}
started=datetime.now(timezone.utc).isoformat()
proc=subprocess.run([sys.executable,'-u','-m','scripts.pde_experiment','run','--scale','212','--device','mps'],cwd=root,env=dict(os.environ,PYTHONPATH='.:src'))
status=dict(returncode=proc.returncode,started_at=started,finished_at=datetime.now(timezone.utc).isoformat(),scale=212,large_started=False)
try:
 if proc.returncode:raise RuntimeError('Formal run stopped with nonzero return code')
 run=root/'experiments/full/212'
 final=json.loads((run/'COMPLETE.json').read_text());assert len(final['results'])==4
 groups=list((run/'cases').glob('*/*/complete.json'));assert len(groups)==32
 rows=[];total=0
 for mark in groups:
  for name,h in json.loads(mark.read_text())['files'].items():assert sha(mark.parent/name)==h
  d=json.loads((mark.parent/'search.json').read_text());assert len(d['attempts'])==2000 and len(d['history'])==5
  total+=len(d['attempts']);last=d['history'][-1]
  from math_tokenizer.eqgpt.evaluation import classify
  case=mark.parent.parent.name;arm=mark.parent.name
  labels=[classify(g,case)['status'] for g in last['gre']]
  rows.append(dict(case=case,arm=arm,top1=labels[0] if labels else 'no_candidate',top10_correct='correct' in labels))
 assert total==64000 and all(sha(Path(p))==h for p,h in mlp.items())
 summary={arm:dict(cases=sum(r['arm']==arm for r in rows),top1_hits=sum(r['arm']==arm and r['top1']=='correct' for r in rows),top10_hits=sum(r['arm']==arm and r['top10_correct'] for r in rows)) for arm in final['results']}
 (report/'summary.json').write_text(json.dumps(dict(rows=rows,aggregate=summary,candidates=total,mlp_unchanged=True),ensure_ascii=False,indent=2)+'\n')
 status.update(status='COMPLETE_VERIFIED',groups=32,candidates=total)
except Exception as e:
 status.update(status='FAILED_OR_INCOMPLETE',error=repr(e));(report/'verification_error.txt').write_text(traceback.format_exc())
finally:
 (report/'exit.json').write_text(json.dumps(status,ensure_ascii=False,indent=2)+'\n')
sys.exit(0 if status.get('status')=='COMPLETE_VERIFIED' else proc.returncode or 1)
