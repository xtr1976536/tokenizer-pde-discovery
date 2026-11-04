"""Three rounds: assets/data; code/results; clean export/GitHub delivery."""
import argparse,ast,csv,gzip,hashlib,json,os,re,subprocess,sys,tempfile
from pathlib import Path
from urllib.parse import unquote
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/repository_audit'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for c in iter(lambda:f.read(8*1024*1024),b''):h.update(c)
 return h.hexdigest()
def run(*args,cwd=ROOT):return subprocess.check_output(args,cwd=cwd,text=True,stderr=subprocess.STDOUT)
def tracked():return [ROOT/p for p in run('git','ls-files','-z').split('\0') if p]
def metrics():
 base=ROOT/'experiments/upstream_current636';n=0
 for p in base.rglob('*.per_formula.jsonl.gz'):
  rows=list(map(json.loads,gzip.open(p,'rt')));d=json.loads(p.with_name(p.name.replace('.per_formula.jsonl.gz','.json')).read_text())
  assert sum(r['initial_vocab_tokens'] for r in rows)/sum(r['final_tokens'] for r in rows)==d['tokenizer compression']
  for key,num,den in [('operation','commands_intact','commands'),('numbers','numbers_intact','numbers')]:
   total=sum(r[den] for r in rows)
   if total:assert abs(100*sum(r[num] for r in rows)/total-d[key])<1e-10
  n+=1
 tests=json.loads((base/'test_results.json').read_text());assert len(tests)==4
 assert len({r['test_ids_sha256'] for r in tests})==1
 assert [r['total_ids'] for r in tests if r['method']=='ast_bpe']==[256]
 return n
def downstream():
 base=ROOT/'experiments/full/212';groups=list((base/'cases').glob('*/*/complete.json'));assert len(groups)==32;count=0
 for p in groups:
  for name,h in json.loads(p.read_text())['files'].items():
   if not name.endswith('.pt'):assert sha(p.parent/name)==h,str(p.parent/name)
  d=json.loads((p.parent/'search.json').read_text());assert len(d['history'])==5 and len(d['attempts'])==2000;count+=len(d['attempts'])
 assert count==64000
 for p in (base/'shared_pretrain').glob('*/training.json'):
  d=json.loads(p.read_text());assert d['epochs']==100 and len(d['history'])==100 and d['samples']==6784
 manifest=json.loads((ROOT/'reports/full212_run/artifacts_manifest.json').read_text())
 for d in manifest['files']:assert sha(ROOT/d['path'])==d['sha256'],d['path']
 return dict(groups=len(groups),candidates=count,raw_artifact_hashes=len(manifest['files']))
def check_links(files):
 bad=[];checked=0
 for p in files:
  if p.suffix!='.md' or p.is_symlink():continue
  # Author/vendor README and historical external corpus narrative remain original snapshots.
  if 'eqgpt_original' in p.parts or 'data' in p.parts or 'corpus_revision' in p.parts:continue
  for link in re.findall(r'\]\(([^)]+)\)',p.read_text()):
   target=link.split('#')[0]
   if not target or '://' in target or target.startswith('mailto:'):continue
   f=p.parent/unquote(target.strip('<>'))
   checked+=1
   if not f.exists():bad.append([str(p.relative_to(ROOT)),link])
 assert not bad,bad
 return checked
def release():
 repo='xtr1976536/tokenizer-pde-discovery';tag='full212-815cc37'
 d=json.loads(run('gh','api',f'repos/{repo}/releases/tags/{tag}'));assert not d['draft']
 pages=json.loads(run('gh','api','--paginate','--slurp',f'repos/{repo}/releases/{d["id"]}/assets?per_page=100'));assets={x['name']:x for page in pages for x in page}
 manifest=json.loads((ROOT/'reports/full212_run/weights_manifest.json').read_text())
 assert set(assets)=={e['asset'] for e in manifest['files']}|{'weights_manifest.json'}
 manifest_path=ROOT/'reports/full212_run/weights_manifest.json'
 assert assets['weights_manifest.json']['digest']=='sha256:'+sha(manifest_path)
 local_verified=0
 for e in manifest['files']:
  a=assets[e['asset']];assert a['state']=='uploaded' and a['size']==e['bytes'] and a['digest']=='sha256:'+e['sha256']
  present=[ROOT/p for p in e['paths'] if (ROOT/p).exists()]
  if present:
   first=present[0];assert sha(first)==e['sha256'];local_verified+=1
   for target in present:
    if not target.samefile(first):assert sha(target)==e['sha256']
 result=dict(status='PASS',public_release=d['html_url'],unique_weights=len(manifest['files']),checkpoint_paths=sum(len(e['paths']) for e in manifest['files']),unique_bytes=manifest['total_unique_bytes'],github_sha256_match=True,local_unique_weights_verified=local_verified,all_local_and_github_sha256_match=local_verified==len(manifest['files']))
 (OUT/'weights_release.json').write_text(json.dumps(result,indent=2)+'\n');return result
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--round',type=int,choices=[1,2,3],required=True);a=ap.parse_args();OUT.mkdir(exist_ok=True);result=dict(round=a.round,status='PASS')
 if a.round==1:
  files=tracked();assert all(p.exists() for p in files)
  assert all(p.resolve().is_relative_to(ROOT) for p in files if p.is_symlink())
  from math_tokenizer.eqgpt.author import verify_source
  verify_source();result['author_source_hashes']='PASS';result['markdown_links']=check_links(files)
  from scripts.pde_experiment import verify
  result['data_scales']={str(n):verify(n)[1]['samples'] for n in [212,636]}
  result['full_dataset_validation']=run(sys.executable,'-m','scripts.check_release','--round','1').strip()
  result['files']=len(files);result['downstream']=downstream()
 elif a.round==2:
  result['pytest']=run(sys.executable,'-m','pytest','-q').strip()
  for p in tracked():
   if p.suffix=='.py' and ('scripts' in p.parts or 'src' in p.parts or 'tests' in p.parts):ast.parse(p.read_text(),filename=str(p))
  result['upstream_metric_outputs']=metrics();result['downstream']=downstream()
  legacy=json.loads((ROOT/'experiments/upstream_current636/legacy_sources.json').read_text())
  for label,expected in legacy['retained_functions_ast_sha256'].items():
   file,name=label.split(':');tree=ast.parse((ROOT/file).read_text());node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name)
   assert hashlib.sha256(ast.dump(node,include_attributes=False).encode()).hexdigest()==expected
  result['legacy_shared_function_hashes']='PASS'
  from math_tokenizer.eqgpt.paired_pipeline import fingerprint
  assert fingerprint()==json.loads((ROOT/'experiments/full/212/RUN.json').read_text())['code']
  result['full212_runtime_fingerprint_preserved']=True
  from scripts.run_current_upstream import prepare,initial
  from scripts.upstream_tokenizers import load
  plans=prepare();result['split']={k:len(v) for k,v in plans.items()}
  for name in ['qwen2.5','llama3','o200k_base']:
   m=load(name);ini=initial(m)
   for text in [r'u_{t}+\sin(u)=0','α+123.45']:
    assert m.decode(m.encode(text))==ini.decode(ini.encode(text))==text
  result['baseline_roundtrip']='PASS'
 else:
  # Export exactly the staged Git snapshot; no current workspace caches or weights.
  with tempfile.TemporaryDirectory(prefix='tokenizer-repository-check-') as tmp:
   index_tree=run('git','write-tree').strip();tar=Path(tmp)/'snapshot.tar'
   with tar.open('wb') as f:subprocess.run(['git','archive',index_tree],cwd=ROOT,stdout=f,check=True)
   checkout=Path(tmp)/'checkout';checkout.mkdir();subprocess.run(['tar','-xf',str(tar),'-C',str(checkout)],check=True);tar.unlink()
   env=dict(os.environ,PYTHONPATH='.:src')
   p=subprocess.run([sys.executable,'-m','pytest','-q'],cwd=checkout,env=env,capture_output=True,text=True);assert p.returncode==0,p.stdout+p.stderr
   result['fresh_export_tests']=p.stdout.strip();result['export_tree']=index_tree
   result['plan']=json.loads(subprocess.check_output([sys.executable,'-m','scripts.pde_experiment','plan'],cwd=checkout,env=env,text=True))['scales']
   code="from scripts.run_current_upstream import prepare; from scripts.upstream_tokenizers import load; p=prepare(); assert len(p['test'])==63; [load(n) for n in ['qwen2.5','llama3','o200k_base']]; print('PASS')"
   subprocess.run([sys.executable,'-c',code],cwd=checkout,env=env,check=True,capture_output=True,text=True)
   result['fresh_export_upstream']='PASS'
   import shutil
   shutil.rmtree(checkout/'experiments/upstream_current636')
   p=subprocess.run([sys.executable,'-m','scripts.run_current_upstream'],cwd=checkout,env=env,capture_output=True,text=True)
   assert p.returncode==0,p.stdout[-1000:]+p.stderr[-2000:]
   actual=json.loads((checkout/'experiments/upstream_current636/test_results.json').read_text())
   expected=json.loads((ROOT/'experiments/upstream_current636/test_results.json').read_text())
   assert len(actual)==len(expected)==4
   for aa,bb in zip(actual,expected):
    assert aa['method']==bb['method'] and aa['test_ids_sha256']==bb['test_ids_sha256']
    for key in ['tokenizer compression','operation','numbers']:assert abs(aa[key]-bb[key])<1e-12
   result['fresh_full_upstream_rerun']='PASS: all four experiments, newly trained vocabularies, identical main metrics'
   code="from scripts.audit_repository import release; r=release(); assert r['local_unique_weights_verified']==0; print('PASS')"
   subprocess.run([sys.executable,'-c',code],cwd=checkout,env=env,check=True,capture_output=True,text=True)
   result['fresh_export_without_local_weights']='PASS: GitHub digest verification'
  result['weights']=release()
  result['remote']=json.loads(run('gh','api','repos/xtr1976536/tokenizer-pde-discovery','--jq','{full_name,private,default_branch}'));assert not result['remote']['private']
 target=OUT/f'round{a.round}.json';tmp=target.with_suffix('.json.tmp')
 tmp.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');os.replace(tmp,target)
 print('PASS',a.round,json.dumps(result,ensure_ascii=False)[:1000])
if __name__=='__main__':main()
