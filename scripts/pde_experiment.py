"""Two-scale212/636 paired experiment; full protocol and explicit small verification."""
import argparse,gzip,json,hashlib,gc,tempfile
from pathlib import Path
from math_tokenizer.corpus import ROOT
from math_tokenizer.eqgpt.study3_corpus import paired_rows,SIZES,ARMS
from math_tokenizer.eqgpt.paired_pipeline import POLICY,fingerprint
from math_tokenizer.eqgpt.expanded_corpus import sha
SETTINGS=dict(seed=525,layers=6,epochs=100,batch_size=128,positions=50,search_limit=48,
 learning_rate=1e-4,cycles=5,attempts=400,continuation_epochs=5,continuation_lr=1e-5,epi=.2)

def dump(p,x):
 p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')

def prepare():
 from math_tokenizer.eqgpt.eqgpt_fast_tokenizers import train_or_load
 from math_tokenizer.eqgpt.eqgpt_fast_config import ALPHABET
 from math_tokenizer.eqgpt.latex_codec import structure_to_latex
 for scale in [212,636]:
  folder=ROOT/'data/formulas'/str(scale)
  rows=[json.loads(l) for l in (folder/'sources.jsonl').read_text().splitlines()];assert len(rows)==scale
  models=train_or_load([structure_to_latex(r['gre_ids']) for r in rows],folder/'tokenizers',SIZES,ALPHABET)
  lengths={a:0 for a in ARMS}
  with gzip.open(folder/'paired.jsonl.gz','wt') as out:
   for r in rows:
    variants=list(paired_rows(r));assert len(variants)==32
    for v in variants:
     lengths['eqgpt_reference']=max(lengths['eqgpt_reference'],len(v['gre_ids']));assert len(v['gre_ids'])<=50
     for (_,size),m in models.items():
      ids=m.encode(v['latex']);assert m.decode(ids)==v['latex'] and len(ids)+2<=50
      lengths[f'ast_bpe_{size}']=max(lengths[f'ast_bpe_{size}'],len(ids)+2)
     out.write(json.dumps(v,ensure_ascii=False)+'\n')
  files={str(p.relative_to(folder)):sha(p) for p in folder.rglob('*') if p.is_file() and p.name!='manifest.json'}
  dump(folder/'manifest.json',dict(scale=scale,samples=scale*32,max_lengths=lengths,files=files))

def verify(scale):
 f=ROOT/'data/formulas'/str(scale);m=json.loads((f/'manifest.json').read_text())
 for name,h in m['files'].items():
  if sha(f/name)!=h:raise ValueError('Data asset changed:'+name)
 return f,m

def run(scale,device,smoke):
 import torch
 from math_tokenizer.eqgpt.models import build_models
 from math_tokenizer.eqgpt.compact_alphabet import CompactAlphabet
 from math_tokenizer.eqgpt.latex_tokenizers import LatexAdapter
 from math_tokenizer.eqgpt.tokenizer_adapters import Adapter
 from math_tokenizer.eqgpt.round1_runtime import train_round1
 from math_tokenizer.eqgpt.study3_search import search
 from math_tokenizer.eqgpt.existing_surrogate import load_existing
 from math_tokenizer.eqgpt.shared8_config import SHARED8_CASES
 from math_tokenizer.eqgpt.author import tensor_hash
 folder,meta=verify(scale);torch.set_num_threads(8)
 cfg=dict(SETTINGS)
 if smoke:cfg.update(epochs=1,cycles=2,attempts=4,continuation_epochs=1)
 dest=ROOT/'experiments'/('smoke' if smoke else 'full')/str(scale)
 identity=dict(settings=cfg,formal_settings=SETTINGS,smoke=smoke,device=device,source_sha256=sha(folder/'manifest.json'),code=fingerprint(),runner=sha(Path(__file__)))
 if (dest/'RUN.json').exists() and json.loads((dest/'RUN.json').read_text())!=identity:raise ValueError('Run identity changed')
 dump(dest/'RUN.json',identity)
 aa={'eqgpt_reference':Adapter('eqgpt'),**{f'ast_bpe_{n}':LatexAdapter('ast_bpe',CompactAlphabet.load(folder/'tokenizers'/f'ast_bpe_{n}.model.json')) for n in SIZES}}
 with gzip.open(folder/'paired.jsonl.gz','rt') as f:rows=[json.loads(l) for l in f]
 if smoke:
  # Span the whole source range, including new formulas in636; not the same first8 for both scales.
  indices=[round(i*(scale-1)/7) for i in range(8)]
  chosen=[rows[i*32] for i in indices]
 else:chosen=rows
 models,initial=build_models(aa,50,525,device,n_layers=6);results={}
 for arm in ARMS:
  model=models.pop(arm);pre=dest/'shared_pretrain'/arm
  encoded=[aa[arm].sequence(r['gre_ids'] if arm=='eqgpt_reference' else r['latex']) for r in chosen]
  tr=train_round1(model,encoded,pre,device,525,smoke=smoke,epochs_override=cfg['epochs'])
  state={k:v.detach().clone() for k,v in model.state_dict().items()}
  for case in SHARED8_CASES[:1] if smoke else SHARED8_CASES:
   model.load_state_dict(state);out=dest/'cases'/case/arm
   if (out/'complete.json').exists():
    for n,h in json.loads((out/'complete.json').read_text())['files'].items():assert sha(out/n)==h
    continue
   cache=ROOT/'data/mlp_cache'/case/'surrogate_cache.pt';before=sha(cache)
   _,net,db,_,_=load_existing(cache.parent,case,device,525,10000)
   report=search(model if arm=='eqgpt_reference' else model.core,aa[arm],net,db,case,out,device,reference=arm=='eqgpt_reference',pretrained_path=pre/'pretrained.pt',settings=cfg)
   assert len(report['attempts'])==cfg['cycles']*cfg['attempts'] and sha(cache)==before
   assert all(h['before_sha256']!=h['after_sha256'] for h in report['history'] if h['continuation_sequences'])
   dump(out/'complete.json',dict(files={p.name:sha(p) for p in out.iterdir() if p.is_file()}))
  results[arm]=dict(parameters=initial[arm]['parameters'],layers=6,trained_samples=len(chosen),pretrain_updated=tr['initial_sha256']!=tr['final_sha256'],search_attempts=cfg['attempts']*cfg['cycles'])
  del model,state;gc.collect()
  if device=='mps':torch.mps.empty_cache()
 dump(dest/'COMPLETE.json',dict(status='PASS',scale=scale,smoke=smoke,results=results,selected_source_ids=[r['source_id'] for r in chosen] if smoke else None))

def main():
 ap=argparse.ArgumentParser();ap.add_argument('phase',choices=['plan','prepare','smoke','run']);ap.add_argument('--scale',type=int,choices=[212,636]);ap.add_argument('--device',choices=['cpu','mps','cuda'],default='mps');a=ap.parse_args()
 if a.phase=='plan':print(json.dumps(dict(scales=[212,636],settings=SETTINGS,pipeline=POLICY),indent=2));return
 if a.phase=='prepare':prepare();return
 for scale in [a.scale] if a.scale else [212,636]:run(scale,a.device,a.phase=='smoke')
if __name__=='__main__':main()
