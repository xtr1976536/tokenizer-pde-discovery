"""Check terminal-policy alignment using the actual author's cleanup function."""
import pytest
from math_tokenizer.eqgpt.author import numerical_namespace
from math_tokenizer.eqgpt.latex_experiment import RewardEvaluator
from math_tokenizer.eqgpt.latex_codec import latex_to_structure


def evaluator():
    e=RewardEvaluator(None,None,'Burgers_equation',['x','t'],'cpu',.2)
    calls=[]
    def reward(ids,*args):
        calls.append(list(ids))
        return .8,{},e.mask
    e.env['calculate_reward']=reward
    return e,calls


def test_cap_without_eos_uses_official_cleanup_and_preserves_raw():
    e,calls=evaluator()
    row=dict(stopped=False,stop_reason='token_limit',ids=[10,11],text='(u_t)+(u)\\cdot(u_x)+(u_xx)+(u_xx)')
    r=e.evaluate(row)
    assert r['status']=='evaluated' and len(calls)==1
    assert r['raw_gre']==latex_to_structure(row['text'])
    expected=[5]+numerical_namespace('cpu','Burgers_equation')['delete_duplicate'](r['raw_gre'][1:].copy())
    assert r['gre']==expected and r['normalized']
    assert r['termination_added_after_parse'] and not r['eos_generated']
    assert r['stopped'] is False and r['ids']==row['ids']
    assert calls[0]==expected[1:]


def test_same_text_eos_and_cap_have_same_scored_structure():
    a,_=evaluator();b,_=evaluator()
    text='(u_t)+(u)\\cdot(u_x)+(u_xx)'
    complete=a.evaluate(dict(stopped=True,stop_reason='eos',text=text))
    capped=b.evaluate(dict(stopped=False,stop_reason='token_limit',text=text))
    assert complete['status']==capped['status']=='evaluated'
    assert complete['gre']==capped['gre'] and complete['reward']==capped['reward']


@pytest.mark.parametrize('text',['(u_t)+','(u_t)+(u','(u_t)='])
def test_broken_prefix_is_not_completed(text):
    e,calls=evaluator()
    r=e.evaluate(dict(stopped=False,stop_reason='token_limit',text=text))
    assert r['status'] in ('syntax_error','unsupported_expression') and not calls


@pytest.mark.parametrize('reason',['unexpected_control','byte_limit'])
def test_non_length_failures_remain_failures(reason):
    e,calls=evaluator()
    r=e.evaluate(dict(stopped=False,stop_reason=reason,text='u_t+u_xx'))
    assert r['status']=='incomplete' and not calls
