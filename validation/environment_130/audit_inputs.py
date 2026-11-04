"""Exhaustive training-input audit plus real longest-example forward/backward.

Does not fit MLP, run PINN, take optimizer steps or alter model checkpoints.
"""
import argparse,ast,collections,gc,gzip,hashlib,json
from pathlib import Path
import torch
from torch import nn
from torch.utils.data import DataLoader
from math_tokenizer.corpus import ROOT
from math_tokenizer.eqgpt.author import training_namespace,tensor_hash
from math_tokenizer.eqgpt.models import build_models
from math_tokenizer.eqgpt.tokenizer_adapters import Adapter
from math_tokenizer.eqgpt.latex_tokenizers import LatexAdapter
from math_tokenizer.eqgpt.compact_alphabet import CompactAlphabet
from math_tokenizer.eqgpt.latex_codec import structure_to_latex
from math_tokenizer.eqgpt.expanded_corpus import sha
from scripts.pde_experiment import verify,effective_settings
SETTINGS=effective_settings(130)


def main():
 ap=argparse.ArgumentParser();ap.add_argument('--device',default='mps',choices=['cpu','mps','cuda']);args=ap.parse_args()
 torch.set_num_threads(8);all_sources={};results={}
 for scale in [212,636]:
  folder,meta=verify(scale)
  sources=[json.loads(l) for l in (folder/'sources.jsonl').read_text().splitlines()]
  with gzip.open(folder/'paired.jsonl.gz','rt') as f:rows=[json.loads(l) for l in f]
  all_sources[scale]=sources
  texts=[structure_to_latex(r['gre_ids']) for r in sources]
  tokenizer_meta=json.loads((folder/'tokenizers/manifest.json').read_text())
  assert tokenizer_meta['corpus_sha256']==hashlib.sha256(json.dumps(texts,ensure_ascii=False).encode()).hexdigest()
  aa={'eqgpt_reference':Adapter('eqgpt'),**{f'ast_bpe_{n}':LatexAdapter('ast_bpe',CompactAlphabet.load(folder/'tokenizers'/f'ast_bpe_{n}.model.json')) for n in [280,315,350]}}
  models,initial=build_models(aa,50,525,args.device,n_layers=6)
  assert len({x['backbone_initial_sha256'] for x in initial.values()})==1
  env=training_namespace(args.device);report={}
  for arm in aa:
   model=models.pop(arm);adapter=aa[arm];encoded=[]
   # Exact same list comprehension as the formal training entry, no filtering.
   for r in rows:
    seq=adapter.sequence(r['gre_ids'] if arm=='eqgpt_reference' else r['latex'])
    assert 2<len(seq)<=50 and seq[0]==adapter.bos and seq[-1]==adapter.eos
    assert all(0<=int(i)<adapter.vocab_size for i in seq)
    assert seq.count(adapter.bos)==seq.count(adapter.eos)==1
    if arm!='eqgpt_reference':assert adapter.decode(seq[1:-1])==r['latex']
    encoded.append(seq)
   assert len(encoded)==len(rows)==scale*32
   # Exercise every teacher-forced batch with the author's real collator.
   loader=DataLoader(env['MyDataSet'](encoded),batch_size=128,collate_fn=env['MyDataSet'](encoded).padding_batch,shuffle=False)
   observed=0;nonpad_targets=0
   for x,y in loader:
    assert x.shape==y.shape and x.shape[1]<=50 and x.shape[0]<=128
    assert int(x.min())>=0 and int(x.max())<adapter.vocab_size
    assert int(y.min())>=0 and int(y.max())<adapter.vocab_size
    observed+=len(x);nonpad_targets+=int((y!=0).sum())
   assert observed==len(rows) and nonpad_targets==sum(len(v)-1 for v in encoded)
   # Real6-layer CE/backward on two longest sequences; no optimizer step.
   longest=sorted(range(len(encoded)),key=lambda i:len(encoded[i]),reverse=True)[:2]
   ds=env['MyDataSet']([encoded[i] for i in longest])
   x,y=ds.padding_batch([ds[i] for i in range(2)])
   before=tensor_hash(model.state_dict());model.train();model.zero_grad(set_to_none=True)
   logits,_=model(x.to(args.device));loss=nn.CrossEntropyLoss(ignore_index=0)(logits,y.to(args.device).reshape(-1))
   assert torch.isfinite(loss);loss.backward()
   assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
   assert any(p.grad is not None and bool(p.grad.abs().sum()>0) for p in model.parameters())
   assert tensor_hash(model.state_dict())==before
   report[arm]=dict(sources=scale,samples=len(encoded),excluded_sources=0,excluded_augmented_samples=0,max_sequence_length=max(map(len,encoded)),vocab_size=adapter.vocab_size,
    batches_per_epoch=len(loader),epochs=130,formal_optimizer_steps=130*len(loader),training_targets_per_epoch=nonpad_targets,forward_backward_loss=float(loss.detach().cpu()),weights_unchanged=True,
    sequence_sha256=hashlib.sha256(json.dumps(encoded).encode()).hexdigest())
   print('PASS',scale,arm,report[arm],flush=True)
   del model,logits,loss,x,y,loader,ds;gc.collect()
   if args.device=='mps':torch.mps.empty_cache()
  results[scale]=report
 assert all_sources[212]==all_sources[636][:212]
 # Actual entry has no PINN call, though source helpers remain available.
 tree=ast.parse((ROOT/'scripts/pde_experiment.py').read_text())
 calls={ast.unparse(n.func) for n in ast.walk(tree) if isinstance(n,ast.Call)}
 assert not any('pinn' in c.lower() for c in calls)
 out=ROOT/'validation/environment_130/fairness';out.mkdir(parents=True,exist_ok=True)
 (out/'audit.json').write_text(json.dumps(dict(status='PASS',device=args.device,settings=SETTINGS,scales=results,source212_nested_exactly=True,
  active_entry_runs_PINN=False,pinn_helpers_retained=True,actual_full_training_performed=False,verification_scope='All inputs encoded and collated; real forward/backward on two longest inputs per arm/scale, no optimizer step',
  limitations=['Equal source records/epochs do not imply equal token targets, FLOPs or parameter counts.','SAT enabled only for EqGPT by user design.','Original212 numerical-token findings remain; this audit proves training admission, not correctness of every numerical derivative.']),ensure_ascii=False,indent=2)+'\n')
 print('ALL FAIRNESS INPUT CHECKS PASS',flush=True)
if __name__=='__main__':main()
