"""Independent audit of term semantics and active reward path; no algorithm changes."""
import collections,itertools,json,math,warnings
from pathlib import Path
import numpy as np,sympy as sp,torch
from torch import nn
from math_tokenizer.corpus import ROOT
from math_tokenizer.eqgpt.author import numerical_namespace,DICTIONARY
from math_tokenizer.eqgpt.corpus_conversion import atoms,U,COORDS
from math_tokenizer.eqgpt.latex_experiment import RewardEvaluator,select_top
from math_tokenizer.eqgpt.study3_search import target_gre
from math_tokenizer.eqgpt.shared8_config import SHARED8_CASES
D=DICTIONARY;W=D['word2id'];WORDS=D['id2word'];X,Y,Z,T=COORDS
OUT=ROOT/'reports/scorer_audit'

def save(name,x):(OUT/name).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')

def atom_tests():
 records=[]
 # Baseline dictionary has51 used atoms; audit every51 noncontrol entries, all dimensions where meaningful.
 for axes in [('x','t'),('t','x'),('x','y','t'),('t','y','x'),('x','y','z','t'),('t','z','y','x')]:
  syms=[dict(zip('xyzt',COORDS))[a] for a in axes]
  points=np.random.default_rng(20261104).uniform(.21,.71,(4,len(axes)))
  class Field(nn.Module):
   def forward(self,v):return torch.exp(torch.sin(v.prod(dim=1,keepdim=True))+v.sum(dim=1,keepdim=True))
  field=sp.exp(sp.sin(sp.prod(syms))+sum(syms));env=numerical_namespace('cpu','Wave_equation')
  for i in range(6,57):
   word=WORDS[i];expr=atoms()[i]
   if word in ['Laplace(u)','Laplace(utt)','BiLaplace(u)'] and 'y' not in axes:
    records.append(dict(word=word,id=i,axes=axes,status='INAPPLICABLE_BY_SPATIAL_DOMAIN'));continue
   if word in ['Laplace(u)','Laplace(utt)','BiLaplace(u)']:
    spatial=[dict(zip('xyzt',COORDS))[a] for a in axes if a!='t']
    base=sp.diff(U,T,2) if word=='Laplace(utt)' else U
    expr=sum(sp.diff(base,a,2) for a in spatial)
    if word=='BiLaplace(u)':expr=sum(sp.diff(expr,a,2) for a in spatial)
   # U's formal args include xyz even when expression does not depend on those derivatives.
   deriv_axes={v for d in expr.atoms(sp.Derivative) for v in d.variables}
   if not deriv_axes<=set(syms):continue
   expected=expr.subs(U,field).doit()
   if not expected.free_symbols<=set(syms):continue
   try:
    coords=torch.tensor(points,dtype=torch.float64,requires_grad=True)
    actual=np.asarray(env['calculate_terms'](word,Field(),coords,list(axes))).reshape(-1)
    expected=expr.subs(U,field).doit();wanted=np.broadcast_to(np.asarray(sp.lambdify(syms,expected,'numpy')(*points.T),dtype=float),actual.shape)
    good=bool(np.isfinite(actual).all() and np.allclose(actual,wanted,rtol=1e-7,atol=1e-8))
    records.append(dict(word=word,id=i,axes=axes,status='PASS' if good else 'MISMATCH',max_abs_error=float(np.max(abs(actual-wanted)))))
   except Exception as e:records.append(dict(word=word,id=i,axes=axes,status='ERROR',error=repr(e)))
 return records


def oracle(gre,values,epi=.2):
 # Explicit left-to-right scalar products/division, independent of author's grouping code.
 terms=[];current=None;op='*'
 for token in gre[1:]:
  if token in (1,2):
   if current is not None:terms.append(current.copy())
   current=None;op='*'
  elif token==3:op='*'
  elif token==4:op='/'
  else:
   v=np.asarray(values[WORDS[token]])
   current=v.copy() if current is None else current*v if op=='*' else current/v
 if not terms:raise ValueError('no terms')
 A=np.column_stack(terms)
 # Exact duplicate numeric columns is an authored policy, reproduce it independently.
 keep=[]
 for j in range(A.shape[1]):
  if not any(np.array_equal(A[:,j],A[:,k]) for k in keep):keep.append(j)
 A=A[:,keep];A=A[np.isfinite(A).all(axis=1)]
 lhs=-A[:,0];coef=np.linalg.lstsq(A[:,1:],lhs,rcond=None)[0];rhs=A[:,1:]@coef
 den=((lhs-lhs.mean())**2).sum()
 if den==0:return None
 r2=1-((lhs-rhs)**2).sum()/den
 return float((1-epi*np.log10(A.shape[1]))*r2)


def reward_tests():
 n=41;x=np.linspace(.2,1.4,n)
 values={'ut':np.sin(x)+.1*x,'ux':np.cos(2*x),'uxx':x*x+.2,'u':np.exp(.3*x)}
 examples=[['ut','+','ux','+','uxx'],['ux','+','ut','+','uxx'],['uxx','+','ux','+','ut'],['ut','+','ux','*','u','+','uxx'],['ut','+','ux','/','u','+','uxx'],['ut','+','ux','/','u','*','uxx']]
 out=[]
 for words in examples:
  gre=[5]+[W[w] for w in words]+[1];env=numerical_namespace('cpu','Wave_equation')
  env['calculate_terms']=lambda word,*args:values[word].copy()
  got,_,_=env['calculate_reward'](gre[1:].copy(),None,None,{},env['get_mask_invalid'](['x','t']),['x','t'],.2)
  expected=oracle(gre,values);assert np.isclose(got,expected,rtol=1e-12,atol=1e-12)
  out.append(dict(words=words,author=float(got),independent=float(expected),matches=True))
 return out


def edge_tests():
 env=numerical_namespace('cpu','Wave_equation')
 def ids(words):return [W[w] for w in words]
 before=ids(['u','*','ux','+','u','/','ux','E'])
 cleaned=env['delete_duplicate'](before.copy())
 result=dict(operator_blind_cleanup=dict(before=[WORDS[i] for i in before],after=[WORDS[i] for i in cleaned]))
 # False coordinate-presence rejection: sin(u) contains the letter x nowhere,
 # while complex derivatives such as u_x certainly do. Preserve source intent vs semantics distinction.
 result['anchor_order_rewards']=reward_tests()[:3]
 # Data NaNs/zero variance should be reported as failure, not silently selected.
 class Dummy(nn.Module):
  def forward(self,z):return (z[:,0:1]**2+z[:,1:2]**2+1)
 for label,val in [('constant',np.ones(8)),('zero',np.zeros(8))]:
  e=RewardEvaluator(Dummy(),torch.ones(8,2,requires_grad=True),'Wave_equation',['x','t'],'cpu',.2)
  e.env['calculate_terms']=lambda word,*args:val.copy()
  g=[5,W['ut'],2,W['uxx'],1]
  with warnings.catch_warnings():
   warnings.simplefilter('ignore');r=e.evaluate(dict(stopped=True,stop_reason='eos',gre=g,text='u_t+u_xx=0'))
  result[label+'_columns']=dict(status=r['status'],reward=r.get('reward'),error=r.get('error'))
 # Fit-equivalent scalar encodings may be treated differently by combination masks.
 x=np.linspace(.2,1.4,41)
 values={'ut':-(x+x*x+x*x*x+np.cos(x)),'ux':np.cos(x),'u':x,'u^2':x*x,'u^3':x*x*x}
 e=numerical_namespace('cpu','Wave_equation');e['calculate_terms']=lambda word,*args:values[word].copy()
 canonical=[5]+ids(['ut','+','ux','+','u','+','u^2','+','u^3'])+[1]
 product=[5]+ids(['ut','+','ux','+','u','+','u','*','u','+','u^3'])+[1]
 scores=[]
 for g in [canonical,product]:
  score,_,_=e['calculate_reward'](g[1:].copy(),None,None,{},e['get_mask_invalid'](['x','t']),['x','t'],.2)
  scores.append(dict(tokens=[WORDS[i] for i in g],author_score=float(score),independent_fit_reward=oracle(g,values)))
 result['combination_zero_representation_effect']=scores
 # One nonfinite division row is removed before regression, not rejected.
 y=x.copy();y[3]=0.
 values={'ut':np.sin(x),'ux':np.cos(x),'u':y,'uxx':x*x+.2}
 e=numerical_namespace('cpu','Wave_equation');e['calculate_terms']=lambda word,*args:values[word].copy()
 g=[5]+ids(['ut','+','ux','/','u','+','uxx'])+[1]
 with warnings.catch_warnings():
  warnings.simplefilter('ignore')
  score,_,_=e['calculate_reward'](g[1:].copy(),None,None,{},e['get_mask_invalid'](['x','t']),['x','t'],.2)
 result['division_nonfinite_row']=dict(input_rows=41,retained_rows=40,author_returns_finite=bool(np.isfinite(score)),reward=float(score),note='Official scorer removes rows with infinite matrix entries; no minimum retained row count is enforced.')
 return result


def real_target_tests():
 from math_tokenizer.eqgpt.existing_surrogate import load_existing
 result=[]
 for case in SHARED8_CASES:
  _,net,db,_,_=load_existing(ROOT/'data/mlp_cache'/case,case,'cpu',525,10000)
  gre=target_gre(case);e=RewardEvaluator(net,db,case,['x','t'],'cpu',.2)
  r=e.evaluate(dict(stopped=True,stop_reason='eos',gre=gre,text='target'))
  vals={WORDS[i]:np.asarray(e.env['calculate_terms'](WORDS[i],net,db,['x','t'])).reshape(-1) for i in set(gre) if i>=6}
  with warnings.catch_warnings():
   warnings.simplefilter('ignore');wanted=oracle(gre,vals)
  match=r['status']=='evaluated' and wanted is not None and np.isclose(r['reward'],wanted,rtol=1e-5,atol=1e-6)
  result.append(dict(case=case,status=r['status'],reward=r.get('reward'),independent_reward=wanted,formula_matches=bool(match)))
 return result


def main():
 OUT.mkdir(exist_ok=True);torch.set_num_threads(4)
 atom=atom_tests();save('atoms.json',dict(records=atom,counts=dict(collections.Counter(r['status'] for r in atom))))
 save('reward_oracle.json',reward_tests());save('edge_cases.json',edge_tests());save('real_targets.json',real_target_tests())
 print('ATOM',collections.Counter(r['status'] for r in atom),'bad',sorted({r['word'] for r in atom if r['status'] in ('MISMATCH','ERROR')}),flush=True)
 print('DONE',flush=True)
if __name__=='__main__':main()
