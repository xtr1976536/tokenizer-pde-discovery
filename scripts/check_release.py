"""Three reproducibility rounds for data, numerical smoke results and release assets."""
import argparse,collections,gzip,json,subprocess,sys,hashlib,platform
from pathlib import Path
from math_tokenizer.corpus import ROOT
from math_tokenizer.eqgpt.expanded_corpus import sha,gre_signature
from math_tokenizer.eqgpt.latex_codec import latex_to_structure,structure_to_latex
from math_tokenizer.eqgpt.corpus_conversion import gre_expression,convert
from math_tokenizer.eqgpt.study3_corpus import paired_rows
from math_tokenizer.eqgpt.compact_alphabet import CompactAlphabet
from math_tokenizer.eqgpt.evaluation import classify
from math_tokenizer.eqgpt.shared8_config import SHARED8_CASES
from scripts.pde_experiment import verify,SETTINGS
from scripts.prepare_downstream_corpus import invariant_support

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--round',type=int,choices=[1,2,3],required=True);a=ap.parse_args()
 out=ROOT/'reports/checks';out.mkdir(exist_ok=True)
 result=dict(round=a.round,status='PASS',scales=[212,636])
 if a.round==1:
  all_sources={};stats={}
  for scale in [212,636]:
   folder,m=verify(scale);sources=[json.loads(l) for l in (folder/'sources.jsonl').read_text().splitlines()];all_sources[scale]=sources
   with gzip.open(folder/'paired.jsonl.gz','rt') as f:rows=[json.loads(l) for l in f]
   assert len(rows)==scale*32
   assert rows==[v for s in sources for v in paired_rows(s)]
   lookup={s['source_id']:s for s in sources};assert len(lookup)==scale
   assert collections.Counter(r['source_id'] for r in rows)==collections.Counter({k:32 for k in lookup})
   tok={n:CompactAlphabet.load(folder/'tokenizers'/f'ast_bpe_{n}.model.json') for n in [280,315,350]}
   for r in rows:
    gre=r['gre_ids'];assert len(gre)<=50 and gre[0]==5 and gre[-1]==1 and gre.count(1)==gre.count(5)==1
    assert gre_signature(gre)==gre_signature(lookup[r['source_id']]['gre_ids']) and latex_to_structure(r['latex'])==gre
    for model in tok.values():
     ids=model.encode(r['latex']);assert model.decode(ids)==r['latex'] and len(ids)+2<=50
   assert all(not any(classify(s['gre_ids'],c)['status']=='correct' for c in SHARED8_CASES) for s in sources)
   stats[scale]=dict(sources=scale,augmented=len(rows),max_lengths=m['max_lengths'])
  assert all_sources[212]==all_sources[636][:212]
  keys=[invariant_support(gre_expression(r['gre_ids'])) for r in all_sources[636]]
  assert len(set(keys[212:]))==424 and not set(keys[:212])&set(keys[212:])
  for r in all_sources[636][212:]:
   assert sha(ROOT/r['source_file'])==r['source_sha256']
   assert convert(r.get('conversion_input',r['original_latex']))['gre_ids']==r['gre_ids']
  result['data']=stats
 elif a.round==2:
  import torch
  from math_tokenizer.eqgpt.author import tensor_hash
  from math_tokenizer.eqgpt.paired_pipeline import fingerprint
  results={}
  for scale in [212,636]:
   f=ROOT/'experiments/smoke'/str(scale);complete=json.loads((f/'COMPLETE.json').read_text());binding=json.loads((f/'RUN.json').read_text())
   assert binding['code']==fingerprint() and binding['runner']==sha(ROOT/'scripts/pde_experiment.py')
   assert binding['formal_settings']==SETTINGS and binding['settings']['epochs']==1
   assert len(complete['selected_source_ids'])==8
   for arm in complete['results']:
    tr=json.loads((f/'shared_pretrain'/arm/'training.json').read_text());assert tr['samples']==8 and tr['epochs']==1 and tr['initial_sha256']!=tr['final_sha256']
    search=f/'cases/Wave_equation'/arm
    for name,h in json.loads((search/'complete.json').read_text())['files'].items():assert sha(search/name)==h
    d=json.loads((search/'search.json').read_text());assert len(d['attempts'])==8 and len(d['history'])==2
    assert all(r['generated_tokens']<=48 for r in d['attempts'])
    for r in d['checkpoints']:
     assert tensor_hash(torch.load(search/r['file'],map_location='cpu',weights_only=True))==r['sha256']
    for h in d['history']:
     assert h['before_sha256']!=h['after_sha256'] if h['continuation_sequences'] else h['before_sha256']==h['after_sha256']
   results[scale]=complete
  assert any(not s.startswith('eqgpt221:') for s in results[636]['selected_source_ids'])
  result['smoke']=results;result['note']='Real six-layer training/search; eight sampled sources each. Entire corpora exercised in round1.'
 else:
  for name in ['README.md','LICENSE','NOTICE.md','pyproject.toml','requirements-runtime.lock.txt']:assert (ROOT/name).exists()
  tests=subprocess.run([sys.executable,'-m','pytest','-q'],cwd=ROOT,capture_output=True,text=True);assert tests.returncode==0,tests.stdout+tests.stderr
  result['tests']=tests.stdout.strip()
  from math_tokenizer.eqgpt.author import verify_source
  verify_source()
  files={str(p.relative_to(ROOT)):sha(p) for p in ROOT.rglob('*') if p.is_file() and not any(x in p.parts for x in ['.git','__pycache__','.pytest_cache','.native_cache','experiments']) and 'reports/checks' not in str(p.relative_to(ROOT))}
  assert not any(p.stat().st_size>=100*1024**2 for p in ROOT.rglob('*') if p.is_file() and 'experiments' not in p.parts)
  result['release_files']=files;result['python']=platform.python_version()
 (out/f'round{a.round}.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');print('PASS round',a.round)
if __name__=='__main__':main()
