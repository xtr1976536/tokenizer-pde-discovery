"""Three independent gates for the full636 import, preserving existing studies."""
import argparse,ast,hashlib,json,math,subprocess,sys,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];REPORT=ROOT/'reports/full636_run';RUN=ROOT/'experiments/full/636'
sys.path.insert(0,str(ROOT))
BASE='a925812fa695b90b0a94ba525d1128522d75b859'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for c in iter(lambda:f.read(8*1024*1024),b''):h.update(c)
 return h.hexdigest()
def load(p):return json.loads(p.read_text(encoding='utf8'))
def gate1():
 m=load(REPORT/'artifacts_manifest.json');weights=load(REPORT/'weights_manifest.json');wm={p:e for e in weights['files'] for p in e['paths']}
 assert len({e['sha256'] for e in weights['files']})==len(weights['files'])
 for e in m['files']:assert sha(ROOT/e['path'])==e['sha256'],e['path']
 if (REPORT/'orchestration_manifest.json').exists():
  for e in load(REPORT/'orchestration_manifest.json')['files']:assert sha(REPORT/e['path'])==e['sha256'],e['path']
 expected={e['path'] for e in m['files']};actual={p.relative_to(ROOT).as_posix() for p in RUN.rglob('*') if p.is_file() and p.suffix!='.pt'}
 assert expected==actual
 groups=list((RUN/'cases').glob('*/*/complete.json'));assert len(groups)==32
 candidates=0;unavailable=[]
 identity=load(RUN/'RUN.json')
 for p in groups:
  marker=load(p)
  for name,h in marker['files'].items():
   rel=(p.parent/name).relative_to(ROOT).as_posix()
   if name.endswith('.pt'):assert wm[rel]['sha256']==h
   else:assert sha(ROOT/rel)==h,rel
  s=load(p.parent/'search.json');assert s['settings']==identity['settings'];assert s['diagnostics_status']=='complete'
  assert len(s['attempts'])==2000 and len(s['history'])==5 and len(s['checkpoints'])==6
  assert len({r['candidate_id'] for r in s['attempts']})==2000
  for i in range(1,6):
   rows=[r for r in s['attempts'] if r['cycle']==i];assert len(rows)==400 and {r['attempt'] for r in rows}==set(range(400))
   assert len(load(p.parent/f'cycle_{i}_trace.json')['candidates'])==400
   h=s['history'][i-1];assert len(h['continuation_losses'])==(5 if h['continuation_sequences'] else 0)
   if h['continuation_sequences']:assert h['before_sha256']!=h['after_sha256']
   if i<5:unavailable.append((p.parent/f'cycle_{i}.pt').relative_to(ROOT).as_posix())
  d=load(p.parent/'diagnostics.json');assert d['truth_not_injected'] and d['executed_after_search']
  candidates+=len(s['attempts'])
 for p in (RUN/'shared_pretrain').glob('*/training.json'):
  d=load(p);assert d['epochs']==100 and d['samples']==20352 and len(d['history'])==100 and d['optimizer_steps']==15900
  assert d['initial_sha256']!=d['final_sha256'] and all(math.isfinite(r['loss']) for r in d['history'])
 return dict(raw_artifact_hashes=len(expected),orchestration_helpers=len(load(REPORT/'orchestration_manifest.json')['files']),case_model_groups=len(groups),candidates=candidates,unique_retained_weights=len(weights['files']),retained_paths=weights['checkpoint_paths'],released_intermediate_paths=len(unavailable),released_intermediates_documented=True)
def gate2(runtime):
 sys.path[:0]=[str(runtime),str(runtime/'src')]
 m=load(REPORT/'runtime_source_manifest.json')
 for e in m['source_files']:assert sha(runtime/e['path'])==e['sha256'],e['path']
 family=set(runtime.glob('src/**/*.py'))|set(runtime.glob('eqgpt_original/code/*.py'))
 fingerprint=hashlib.sha256(json.dumps({p.relative_to(runtime).as_posix():sha(p) for p in family},sort_keys=True).encode()).hexdigest()
 identity=load(RUN/'RUN.json');assert identity['code']==fingerprint;assert identity['runner']==sha(runtime/'scripts/pde_experiment.py')
 assert identity['settings']==load(ROOT/'experiments/full/212/RUN.json')['settings']
 assert identity['source_sha256']==sha(runtime/'data/formulas/636/manifest.json')
 for name,h in load(runtime/'data/formulas/636/manifest.json')['files'].items():assert sha(runtime/'data/formulas/636'/name)==h
 from math_tokenizer.eqgpt.evaluation import classify
 summary=load(REPORT/'summary.json');result={}
 for arm,model in summary['models'].items():
  counts=[0,0]
  for case,expected in model['cases'].items():
   s=load(RUN/'cases'/case/arm/'search.json');gre=s['history'][-1]['gre'];top1=bool(gre) and classify(gre[0],case)['status']=='correct';top10=any(classify(g,case)['status']=='correct' for g in gre)
   assert top1==expected['top1_correct'] and top10==expected['top10_correct'];counts[0]+=top1;counts[1]+=top10
  assert counts==[model['top1_correct'],model['top10_correct']];result[arm]=dict(top1=counts[0],top10=counts[1])
 # Compare the source of models, shared scoring and sampling constants with the small run base.
 critical=['src/math_tokenizer/eqgpt/models.py','src/math_tokenizer/eqgpt/scoring_fixes.py','src/math_tokenizer/eqgpt/study3_corpus.py','src/math_tokenizer/eqgpt/tokenizer_adapters.py','src/math_tokenizer/hard_ast_2026.py','native/hard_ranked_2026.c']
 for rel in critical:
  b=subprocess.check_output(['git','show',m['base_commit']+':'+rel],cwd=ROOT).replace(b'\r\n',b'\n');assert hashlib.sha256(b).hexdigest()==sha(runtime/rel),rel
 tests=REPORT/'tests.log';assert tests.exists() and '47 passed' in tests.read_text(encoding='utf8')
 return dict(runtime_fingerprint=fingerprint,verified_runtime_source_files=len(m['source_files']),critical_model_scoring_tokenizer_files_unchanged=critical,small_and_large_settings_identical=True,recomputed_results=result,regression_tests='47 passed',tests_log_sha256=sha(tests))
def gate3(remote=False):
 modified=subprocess.check_output(['git','diff','--name-only',BASE],cwd=ROOT,text=True).splitlines()
 existing=subprocess.check_output(['git','diff','--name-only','--diff-filter=M',BASE],cwd=ROOT,text=True).splitlines()
 assert not subprocess.check_output(['git','diff','--name-only','--diff-filter=D',BASE],cwd=ROOT)
 allowed={'README.md','ENVIRONMENT.md','reports/README.md','reports/repository_audit/README.md','.gitattributes'}
 protected=[p for p in modified if not (p in allowed or p.startswith(('reports/full636_run/','experiments/full/636/')) or p in {'scripts/audit_full636_archive.py','scripts/prepare_full636_runtime.py','scripts/full636_release.py','scripts/download_full636_weights.py','scripts/test_full636_runtime.py'})]
 assert not protected,protected
 oldattrs=subprocess.check_output(['git','show',BASE+':.gitattributes'],cwd=ROOT).decode().replace('\r\n','\n')
 newattrs=(ROOT/'.gitattributes').read_text();assert newattrs.startswith(oldattrs)
 assert newattrs[len(oldattrs):].strip()=='# Preserve the original full636 archive bytes and their SHA-256 manifests.\nexperiments/full/636/** -text -whitespace\nreports/full636_run/** -text -whitespace'
 for rel in ['src','native','data','eqgpt_original','experiments/full/212','experiments/upstream_current636','docs','visualization']:
  assert not subprocess.check_output(['git','diff',BASE,'--',rel],cwd=ROOT),rel
 links=0;bad=[]
 for p in [ROOT/'README.md',ROOT/'ENVIRONMENT.md',ROOT/'reports/README.md',ROOT/'reports/repository_audit/README.md',*REPORT.glob('*.md')]:
  for target in re.findall(r'\]\(([^)]+)\)',p.read_text(encoding='utf8')):
   target=target.split('#')[0]
   if not target or '://' in target:continue
   links+=1
   if not (p.parent/target).exists():bad.append(str(p.relative_to(ROOT))+': '+target)
 assert not bad,bad
 if remote:
  from scripts.full636_release import auth,api,assets,matches
  m=load(REPORT/'weights_manifest.json');h=auth();r=api(h,'/releases/tags/'+m['tag']);assert not r['draft'];aa=assets(h,r['id'])
  assert set(aa)=={e['asset'] for e in m['files'] if e['tag']==m['tag']}|{'weights_manifest.json'}
  for e in m['files']:assert matches(aa[e['asset']],e)
  assert aa['weights_manifest.json']['digest']=='sha256:'+sha(REPORT/'weights_manifest.json')
  release=r['html_url']
  small=load(ROOT/'reports/full212_run/weights_manifest.json');sr=api(h,'/releases/tags/'+small['tag']);assert not sr['draft'];sa=assets(h,sr['id'])
  assert set(sa)=={e['asset'] for e in small['files']}|{'weights_manifest.json'}
  for e in small['files']:assert matches(sa[e['asset']],e)
  original_small_manifest=subprocess.check_output(['git','show',BASE+':reports/full212_run/weights_manifest.json'],cwd=ROOT)
  assert sa['weights_manifest.json']['digest']=='sha256:'+hashlib.sha256(original_small_manifest).hexdigest()
 else:release=None
 return dict(existing_protected_trees_unchanged=True,modified_existing_files=existing,new_files=len(modified)-len(existing),no_existing_files_deleted=True,markdown_local_links=links,public_release=release,remote_assets_verified=remote,existing_full212_release_assets_preserved=remote)
def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--round',type=int,choices=[1,2,3],required=True);ap.add_argument('--runtime',type=Path);ap.add_argument('--remote',action='store_true');a=ap.parse_args()
 if a.round==2 and not a.runtime:ap.error('--runtime is required for round 2')
 result=gate1() if a.round==1 else gate2(a.runtime.resolve()) if a.round==2 else gate3(a.remote)
 d=dict(round=a.round,status='PASS',**result);(REPORT/f'audit_round{a.round}.json').write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf8');print(json.dumps(d,ensure_ascii=False))
if __name__=='__main__':main()
