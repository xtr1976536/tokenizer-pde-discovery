"""Legacy vocabulary-use diagnostics on the frozen current 8:1:1 protocols."""
import csv,json
from pathlib import Path
from math_tokenizer.corpus import ROOT
from math_tokenizer.eqgpt.compact_alphabet import CompactAlphabet
from scripts.evaluate_screened_pde import token_coverage
BASE=ROOT/'experiments/upstream_current636'
def write(path,rows):
 with path.open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
def main():
 records={r['id']:r for r in json.loads((BASE/'input_records.json').read_text())};all_rows=[];summaries=[]
 for protocol in ['capacity','simple_to_complex']:
  folder=BASE/protocol;parts=json.loads((folder/'splits.json').read_text());plans={k:[records[i] for i in ids] for k,ids in parts.items()}
  for file in sorted(folder.glob('ast_bpe_*.model.json')):
   size=int(file.name.split('_')[-1].split('.')[0]);m=CompactAlphabet.load(file)
   train,tf,tp=token_coverage(m,plans['train']);valid,vf,vp=token_coverage(m,plans['validation']);test,hf,hp=token_coverage(m,plans['test'])
   rows=[]
   for i,b in enumerate(m.vocab):
    rows.append(dict(protocol=protocol,total_ids=size,content_id=i,type='base_character' if len(b)==1 else 'learned_merge',text=b.decode('utf-8','backslashreplace'),hex=b.hex(),bytes=len(b),train_count=train[i],train_probability=train[i]/sum(train.values()),train_formulas=tf[i],train_papers=tp.get(i,0),validation_count=valid[i],validation_probability=valid[i]/sum(valid.values()),test_count=test[i],test_probability=test[i]/sum(test.values()),test_formulas=hf[i],test_papers=hp.get(i,0)))
   learned=[r for r in rows if r['type']=='learned_merge'];rare=[r for r in learned if r['train_count']<=10];single=[r for r in learned if r['train_papers']<=1]
   summaries.append(dict(protocol=protocol,total_ids=size,folds=1,base_character_ids=sum(r['type']=='base_character' for r in rows),command_atom_ids=0,learned_merge_ids=len(learned),learned_zero_train_ids=sum(r['train_count']==0 for r in learned),learned_zero_test_ids=sum(r['test_count']==0 for r in learned),learned_train_le_2_ids=sum(r['train_count']<=2 for r in learned),learned_train_le_5_ids=sum(r['train_count']<=5 for r in learned),learned_train_le_10_ids=len(rare),learned_single_paper_ids=len(single),learned_long_rare_ids=sum(r['bytes']>=16 and r['train_count']<=10 for r in learned),max_learned_bytes=max((r['bytes'] for r in learned),default=0),learned_train_le_10_token_mass=sum(r['train_count'] for r in rare)/sum(train.values()),learned_single_paper_token_mass=sum(r['train_count'] for r in single)/sum(train.values()),learned_test_zero_train_ids=sum(r['train_count']==0 and r['test_count']>0 for r in learned),train_token_total=sum(train.values()),test_token_total=sum(test.values())))
   assert sum(r['train_count'] for r in rows)==sum(train.values()) and len(rows)+3==size
   write(folder/f'ast_bpe_{size}.tokens.csv',rows);all_rows+=rows
 out=BASE/'vocabulary_usage';out.mkdir(exist_ok=True)
 write(out/'per_token_usage.csv',all_rows);write(out/'usage_summary.csv',summaries)
 flagged=[r for r in all_rows if r['type']=='learned_merge' and (r['train_count']<=10 or r['train_papers']<=1 or r['bytes']>=16)]
 write(out/'review_learned_merges.csv',flagged)
 (out/'audit.json').write_text(json.dumps(dict(status='PASS',protocols=2,vocabulary_settings=len(summaries),token_rows=len(all_rows),flagged_rows=len(flagged),definitions='same legacy frequency/rarity/length/source-count diagnostics, one frozen split instead of three folds'),indent=2)+'\n')
 print('USAGE_COMPLETE',len(summaries),len(all_rows),flush=True)
if __name__=='__main__':main()
