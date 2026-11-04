"""Current 636 benchmark, legacy tokenizers/protocols, initial/final token CR."""
import gzip,hashlib,json,re,time,csv
from collections import Counter
from pathlib import Path
from math_tokenizer.corpus import ROOT
from math_tokenizer.structure_v2 import complexity
from math_tokenizer.pde_protocol import split_corpus
from math_tokenizer.eqgpt.latex_codec import structure_to_latex
from math_tokenizer.eqgpt.compact_alphabet import CompactAlphabet,train_compact
from math_tokenizer.eqgpt.latex_tokenizers import prefix_model
from math_tokenizer.hard_ast_2026 import HardASTBPE
from math_tokenizer.baselines import ByteTokenizer
from scripts.evaluate_screened_pde import measure,token_coverage
from scripts.run_upstream_811_legacy import alphabet_protocols,SIZES
from scripts.当前四分词器 import load
OUT=ROOT/'experiments/upstream_current636'
SWEEP=[200,224,240,256,272,288,300,320,352,384,416,450]
ALPHABET=list(range(32,127))+[9,10,13]
def sha(x):return hashlib.sha256(x).hexdigest()
def dump(p,x):
 p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def initial(model):
 if isinstance(model,CompactAlphabet):return CompactAlphabet('ast_bpe',model.alphabet,HardASTBPE([bytes([i]) for i in range(256)],[]))
 # All three legacy fixed references begin with the full 256 singleton bytes.
 tok=model.tokenizer
 if hasattr(tok,'_mergeable_ranks'):
  assert all(bytes([i]) in tok._mergeable_ranks for i in range(256))
 else:
  from tokenizers.pre_tokenizers import ByteLevel
  vocab=tok.get_vocab();assert all(c in vocab for c in ByteLevel.alphabet())
 return ByteTokenizer()
def evaluate(model,rows,path):
 supported=[r for r in rows if not hasattr(model,'alphabet') or set(r['latex'].encode())<=model.alphabet]
 if not supported:return dict(full_coverage=False,requested_formulas=len(rows),covered_formulas=0,oov_formulas=len(rows),status='no_covered_formulas')
 summary,details,usage=measure(model,supported)
 init=initial(model)
 for r,d in zip(supported,details):
  ids=init.encode(r['latex']);assert init.decode(ids)==r['latex']
  d['initial_vocab_tokens']=len(ids)
 numerator=sum(d['initial_vocab_tokens'] for d in details);den=sum(d['final_tokens'] for d in details)
 # Remove the historical byte-based primary name; retain explicit source-byte diagnostic.
 summary['source_bytes_per_final_token']=summary.pop('compression_ratio')
 summary.pop('method_initial_compression_ratio',None)
 summary.update({'tokenizer compression':numerator/den,'operation':summary['cmd_pct'],'numbers':summary['num_pct'],
 'initial_vocab_tokens':numerator,'final_vocab_tokens':den,'compression_definition':'sum(actual initial-vocabulary encoding lengths) / sum(actual final-vocabulary encoding lengths)',
 'initial_vocab_content_ids':len(init.vocab) if hasattr(init,'vocab') else init.vocab_size,
 'requested_formulas':len(rows),'covered_formulas':len(supported),'oov_formulas':len(rows)-len(supported),'full_coverage':len(rows)==len(supported)})
 summary['counts']['initial_vocab_tokens']=numerator
 with gzip.open(path.with_suffix('.per_formula.jsonl.gz'),'wt') as f:
  for d in details:f.write(json.dumps(d)+'\n')
 dump(path,summary)
 # Independent aggregation over persisted details.
 replay=list(map(json.loads,gzip.open(path.with_suffix('.per_formula.jsonl.gz'),'rt')))
 assert sum(d['initial_vocab_tokens'] for d in replay)/sum(d['final_tokens'] for d in replay)==summary['tokenizer compression']
 for key,num,de in [('operation','commands_intact','commands'),('numbers','numbers_intact','numbers')]:
  n=sum(d[de] for d in replay)
  if n:assert abs(100*sum(d[num] for d in replay)/n-summary[key])<1e-10
 return summary
def prepare():
 source=ROOT/'data/formulas/636/sources.jsonl';rows=[]
 for r in map(json.loads,source.read_text().splitlines()):
  t=structure_to_latex(r['gre_ids']);out=[];symbols={}
  for v in re.findall(r'\\[A-Za-z]+|\\[^A-Za-z]|[0-9]+(?:\.[0-9]+)?|[^\s]',t):
   if re.fullmatch('[A-Za-z]',v):symbols.setdefault(v,len(symbols));out.append('S'+str(symbols[v]))
   elif v[0:1].isdigit():out.append('N')
   else:out.append(v)
  rows.append(dict(id=r['source_id'],latex=t,source_group=r['source_group'],template_sha256=sha(''.join(out).encode()),canonical_sha256=sha(re.sub(r'\s+','',t).encode()),complexity=complexity(t)))
 for i,r in enumerate(sorted(rows,key=lambda r:(r['complexity']['score'],r['id']))):r['complexity_bin']='simple' if i<len(rows)*.3 else 'moderate' if i<len(rows)*.7 else 'complex'
 plans,split=split_corpus(rows)
 protocol=dict(version='current636-upstream-v1',source_sha256=sha(source.read_bytes()),source_formulas=636,representation='unaugmented structure_to_latex(gre_ids)',split=split,legacy_repository_commit='abff7a367b7a94f8814bf819ac54800e7cc62fdc',sizes=SIZES,capacity_sweep=SWEEP,baseline_names=['qwen2.5','llama3','o200k_base'],main_selection='validation full-coverage Pareto plus ascii98/256',initial_vocab='Hard: singleton declared byte alphabet with no merges; fixed references: verified singleton 256-byte vocabulary before merges',compression='actual initial vocabulary tokens / actual final vocabulary tokens',simple_complex='same 8:1:1 source/template split; simple training subset, complex validation/test subsets; complexity-v2 score ranked 30/40/30 as historical corpus construction',no_three_fold=True)
 if (OUT/'protocol.json').exists():assert json.loads((OUT/'protocol.json').read_text())==protocol
 dump(OUT/'protocol.json',protocol);dump(OUT/'input_records.json',rows)
 return plans
def hard(train,alphabet,size,path):
 if path.exists():return CompactAlphabet.load(path)
 model=train_compact('ast_bpe',[r['latex'] for r in train],size,alphabet);model.save(path);return model
def main():
 plans=prepare();train=plans['train'];validation=plans['validation'];test=plans['test'];results=[]
 schemes=alphabet_protocols(train)
 for name,config in schemes.items():
  alphabet=config['alphabet'];missing=sorted(set().union(*(set(r['latex'].encode()) for r in train))-set(alphabet))
  valid=[s for s in SIZES if s>=len(alphabet)+3]
  if missing:
   results.extend(dict(alphabet=name,total_ids=s,status='training_characters_not_covered',missing_bytes=missing) for s in SIZES);continue
  folder=OUT/'matrix'/name;folder.mkdir(parents=True,exist_ok=True)
  parent=hard(train,alphabet,max(valid),folder/'parent.model.json')
  for size in SIZES:
   if size not in valid:results.append(dict(alphabet=name,total_ids=size,status='below_base_plus_controls'));continue
   model=CompactAlphabet('ast_bpe',alphabet,prefix_model('ast_bpe',parent.internal,size+256-len(alphabet)))
   assert len(model.vocab)+3==size
   model.save(folder/f'{size}.model.json')
   val=evaluate(model,validation,folder/f'{size}.validation.json');tr=evaluate(model,train,folder/f'{size}.train.json')
   row=dict(alphabet=name,total_ids=size,status='evaluated',validation=val,train=tr);results.append(row)
   print('MATRIX',name,size,val['tokenizer compression'],flush=True)
 dump(OUT/'validation_results.json',results)
 metrics=['tokenizer compression','operation','numbers'];eligible=[r for r in results if r['status']=='evaluated' and r['validation']['full_coverage']]
 def dominates(a,b):return all(a['validation'][k]>=b['validation'][k] for k in metrics) and any(a['validation'][k]>b['validation'][k] for k in metrics)
 selected=[r for r in eligible if not any(dominates(a,r) for a in eligible) or (r['alphabet']=='ascii98' and r['total_ids']==256)]
 dump(OUT/'selection.json',[dict(alphabet=r['alphabet'],total_ids=r['total_ids']) for r in selected])
 tests=[]
 for r in selected:
  folder=OUT/'matrix'/r['alphabet'];m=CompactAlphabet.load(folder/f"{r['total_ids']}.model.json")
  tests.append(dict(method='ast_bpe',alphabet=r['alphabet'],total_ids=r['total_ids'],**evaluate(m,test,folder/f"{r['total_ids']}.test.json")))
 baseline={name:load(name) for name in ['qwen2.5','llama3','o200k_base']}
 for name,m in baseline.items():
  evaluate(m,validation,OUT/f'{name}.validation.json');tests.append(dict(method=name,total_ids=m.vocab_size,**evaluate(m,test,OUT/f'{name}.test.json')))
 dump(OUT/'test_results.json',tests)
 # Fixed predeclared old capacity grid: all results reported, no test-based selection.
 for protocol,fit,val,held in [('capacity',train,validation,test),('simple_to_complex',[r for r in train if r['complexity_bin']=='simple'],[r for r in validation if r['complexity_bin']=='complex'],[r for r in test if r['complexity_bin']=='complex'])]:
  assert fit and val and held
  folder=OUT/protocol;folder.mkdir(exist_ok=True);dump(folder/'splits.json',{k:[r['id'] for r in rr] for k,rr in [('train',fit),('validation',val),('test',held)]})
  for size in SWEEP:
   m=hard(fit,ALPHABET,size,folder/f'ast_bpe_{size}.model.json')
   for label,rr in [('train',fit),('validation',val),('test',held)]:evaluate(m,rr,folder/f'ast_bpe_{size}.{label}.json')
   counts,formulas,papers=token_coverage(m,fit);tc,tf,tp=token_coverage(m,held)
   with (folder/f'ast_bpe_{size}.tokens.csv').open('w',newline='') as f:
    w=csv.writer(f,lineterminator='\n');w.writerow(['content_id','type','hex','bytes','train_count','train_formulas','train_sources','test_count','test_formulas','test_sources'])
    for i,b in enumerate(m.vocab):w.writerow([i,'base_character' if len(b)==1 else 'learned_merge',b.hex(),len(b),counts[i],formulas[i],papers.get(i,0),tc[i],tf[i],tp.get(i,0)])
   dump(folder/f'ast_bpe_{size}.usage.json',dict(total_ids=size,active_train_ids=len(counts),active_test_ids=len(tc),unused_content_ids=len(m.vocab)-len(counts),learned_zero_train=sum(len(b)>1 and counts[i]==0 for i,b in enumerate(m.vocab)),learned_low_frequency_le10=sum(len(b)>1 and counts[i]<=10 for i,b in enumerate(m.vocab))))
   print(protocol,size,'COMPLETE',flush=True)
  if protocol=='simple_to_complex':
   for name,m in baseline.items():evaluate(m,val,folder/f'{name}.validation.json');evaluate(m,held,folder/f'{name}.test.json')
 summaries=[]
 for path in OUT.rglob('*.json'):
  d=json.loads(path.read_text())
  if isinstance(d,dict) and 'tokenizer compression' in d:
   summaries.append(dict(file=str(path.relative_to(OUT)),initial_vocab_tokens=d['initial_vocab_tokens'],final_vocab_tokens=d['final_vocab_tokens'],compression=d['tokenizer compression'],operation=d['operation'],numbers=d['numbers'],formulas=d['covered_formulas']))
 with (OUT/'three_metrics.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(summaries[0]),lineterminator='\n');w.writeheader();w.writerows(sorted(summaries,key=lambda r:r['file']))
 from scripts.analyze_current_upstream_usage import main as analyze_usage
 analyze_usage()
 dump(OUT/'COMPLETE.json',dict(status='PASS',scales=[636],experiments=['811_main_and_alphabet_capacity_matrix','simple_to_complex','predeclared_capacity_sweep','vocabulary_usage'],three_fold=False,split_counts={k:len(v) for k,v in plans.items()},matrix_rows=len(results),selected_hard_test_arms=len(selected),numerical_metric_note='All rendered numeric lexemes are single-digit; numbers is degenerate at 100%',transformer_or_search_started=False))
 print('COMPLETE',flush=True)
if __name__=='__main__':main()
