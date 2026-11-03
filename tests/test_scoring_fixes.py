import ast
import copy
import itertools
import numpy as np
import pytest
import sympy as sp
import torch
from torch import nn
from math_tokenizer.eqgpt.author import numerical_namespace, source_tree, DICTIONARY, verify_source
from math_tokenizer.eqgpt.corpus_conversion import atoms, U, COORDS
from math_tokenizer.eqgpt.latex_experiment import RewardEvaluator, select_top
from math_tokenizer.eqgpt.scoring_fixes import delete_duplicate, VERSION

W=DICTIONARY['word2id']
def ids(*words):return [W[w] for w in words]


def test_product_and_quotient_remain_distinct():
    seq=ids('u','*','ux','+','u','/','ux','E')
    assert delete_duplicate(seq)==seq


def test_dedup_keeps_denominator_role_and_factor_multiplicity():
    a=ids('u','*','ux','/','uxx')
    b=ids('ux','/','uxx','*','u')
    c=ids('u','/','ux','*','uxx')
    d=ids('u','*','u','*','ux','/','uxx')
    original=a+[2]+b+[2]+c+[2]+d+[1]
    assert delete_duplicate(original)==a+[2]+c+[2]+d+[1]
    assert original[-1]==1  # original input not modified


@pytest.mark.parametrize('axes',[['x','t'],['t','x'],['x','y','t'],['t','y','x'],['x','y','z','t'],['t','z','y','x']])
def test_corrected_terms_against_independent_symbolic_derivatives(axes):
    symbols=[dict(zip('xyzt',COORDS))[a] for a in axes]
    points=np.random.default_rng(314).uniform(.2,.7,(3,len(axes)))
    class Field(nn.Module):
        def forward(self,x):return torch.exp(torch.sin(x.prod(dim=1,keepdim=True))+x.sum(dim=1,keepdim=True))
    field=sp.exp(sp.sin(sp.prod(symbols))+sum(symbols))
    env=numerical_namespace('cpu','Wave_equation')
    expressions={'sint':sp.sin(COORDS[3]),'(u(u^2)xx)xx':atoms()[56]}
    if 'y' in axes:expressions['uyyt']=atoms()[44]
    spatial=[dict(zip('xyzt',COORDS))[a] for a in axes if a!='t']
    laplace=sum(sp.diff(U,a,2) for a in spatial)
    expressions['BiLaplace(u)']=sum(sp.diff(laplace,a,2) for a in spatial)
    for word,e in expressions.items():
        x=torch.tensor(points,dtype=torch.float64,requires_grad=True)
        got=np.asarray(env['calculate_terms'](word,Field(),x,axes)).reshape(-1)
        expected=sp.lambdify(symbols,e.subs(U,field).doit(),'numpy')(*points.T)
        assert np.allclose(got,expected,rtol=1e-7,atol=1e-8),word


def legacy_env():
    # Recreate a separate actual legacy namespace; function.__globals__ must not
    # point to the corrected namespace when checking the comparison.
    env=numerical_namespace('cpu','Wave_equation')
    tree=source_tree('continue_train_GPT.py')
    nodes=[copy.deepcopy(n) for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ('calculate_reward','delete_duplicate')]
    exec(compile(ast.Module(body=nodes,type_ignores=[]),'<legacy scorer>', 'exec'),env)
    env['calculate_terms']=env['_scoring_original_calculate_terms']
    return env


def test_removal_changes_only_one_top_level_reward_veto():
    # Match the installed code body to the exact author AST minus one statement.
    env=numerical_namespace('cpu','Wave_equation')
    assert env['scoring_version']==VERSION
    assert env['calculate_reward'].__code__.co_filename.endswith(':scoring_fixes')
    verify_source()


def test_equivalent_square_encodings_same_reward_without_combo_veto():
    x=np.linspace(.2,1.4,41)
    values={'ut':-(x+x*x+x*x*x+np.cos(x)),'ux':np.cos(x),'u':x,'u^2':x*x,'u^3':x*x*x}
    a=ids('ut','+','ux','+','u','+','u^2','+','u^3','E')
    b=ids('ut','+','ux','+','u','+','u','*','u','+','u^3','E')
    def score(env,seq):
        env['calculate_terms']=lambda word,*args:values[word].copy()
        return env['calculate_reward'](seq.copy(),None,None,{},env['get_mask_invalid'](['x','t']),['x','t'],.2)[0]
    legacy_a=score(legacy_env(),a)
    corrected=[score(numerical_namespace('cpu','Wave_equation'),s) for s in (a,b)]
    assert legacy_a==0
    assert corrected[0]>0 and corrected[0]==pytest.approx(corrected[1],abs=1e-12)


@pytest.mark.parametrize('words',[
    ('ut','+','ux','+','uxx','E'),('uxx','+','ux','+','ut','E'),
    ('ut','+','ux','*','u','+','uxx','E'),('ut','+','ux','/','u','+','uxx','E')])
def test_unaffected_reward_paths_unchanged(words):
    x=np.linspace(.2,1.4,41)
    values={'ut':np.sin(x)+.1*x,'ux':np.cos(2*x),'uxx':x*x+.2,'u':np.exp(.3*x)}
    scores=[]
    for env in [legacy_env(),numerical_namespace('cpu','Wave_equation')]:
        env['calculate_terms']=lambda word,*args:values[word].copy()
        scores.append(env['calculate_reward'](ids(*words),None,None,{},env['get_mask_invalid'](['x','t']),['x','t'],.2)[0])
    assert scores[0]==pytest.approx(scores[1],abs=1e-12)


def test_gre_and_hard_use_same_fixed_scorer():
    text=r'(u_t)+(u_x)+(u)+(u)\cdot(u)+(u^3)'
    from math_tokenizer.eqgpt.latex_codec import latex_to_structure
    gre=latex_to_structure(text)
    x=np.linspace(.2,1.4,41)
    values={'ut':-(x+x*x+x*x*x+np.cos(x)),'ux':np.cos(x),'u':x,'u^2':x*x,'u^3':x*x*x}
    out=[]
    for reference in (False,True):
        e=RewardEvaluator(None,None,'Wave_equation',['x','t'],'cpu',.2)
        e.env['calculate_terms']=lambda word,*args:values[word].copy()
        row=dict(stopped=True,stop_reason='eos',text=text)
        if reference:row['gre']=gre
        r=e.evaluate(row);assert r['status']=='evaluated'
        assert r['scoring_version']==VERSION
        out.append(r['reward'])
    assert out[0]==pytest.approx(out[1],abs=1e-12)
