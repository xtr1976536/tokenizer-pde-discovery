"""Verify matched sampling, scoring order, SAT isolation and length semantics."""
from types import SimpleNamespace
import torch
from math_tokenizer.eqgpt.paired_pipeline import sample_and_evaluate
from math_tokenizer.eqgpt.author import numerical_namespace
from math_tokenizer.eqgpt.latex_experiment import RewardEvaluator
from math_tokenizer.eqgpt.latex_codec import latex_to_structure

class Core(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.logits=torch.nn.Parameter(torch.zeros(57))
        self.decoder=SimpleNamespace(pos_emb=SimpleNamespace(num_embeddings=10))
    def forward(self,ids):return self.logits.expand(ids.numel(),-1),None

class Adapter:
    bos=5;pad=0;eos=1;vocab_size=57
    def decode(self,ids):return 'u_t+u_xx'

class Recorder:
    def __init__(self):self.mask=torch.ones(57);self.rows=[]
    def evaluate(self,row):self.rows.append(row);return row


def test_sat_changes_only_reference_allowed_ids_and_both_keep_budget():
    for reference in (False,True):
        e=Recorder();e.mask[6]=0
        rows=sample_and_evaluate(Core(),Adapter(),e,40,7,525,reference)
        assert len(rows)==40 and len(e.rows)==40
        assert all(r['generated_tokens']<=7 for r in rows)
        if reference:
            for r in rows:
                for i,t in enumerate(r['ids'][1:]):
                    assert t!=6
                    assert t>=6 if i%2==0 else 1<=t<=4
        else:
            # No control/term/operator mask applied to Hard.
            assert any(0 in r['ids'] or 5 in r['ids'] for r in rows)
            assert any(6 in r['ids'] for r in rows)


def test_each_attempt_has_same_prefix_across_caps():
    model=Core()
    short=sample_and_evaluate(model,Adapter(),Recorder(),20,3,525,False)
    long=sample_and_evaluate(model,Adapter(),Recorder(),20,7,525,False)
    for a,b in zip(short,long):assert a['ids']==b['ids'][:len(a['ids'])]


def test_next_candidate_observes_immediate_score_feedback():
    class Feedback(Recorder):
        def evaluate(self,row):
            result=super().evaluate(row)
            self.mask[6:]=0;self.mask[8]=1
            return result
    rows=sample_and_evaluate(Core(),Adapter(),Feedback(),3,1,525,True)
    assert rows[1]['ids']==[5,8] and rows[2]['ids']==[5,8]


def test_gre_and_latex_share_normalization_status_and_reward():
    text='(u_t)+(u_xx)+(u_xx)'
    raw=latex_to_structure(text)
    def evaluate(reference):
        e=RewardEvaluator(None,None,'Wave_equation',['x','t'],'cpu',.2)
        seen=[]
        def score(ids,*args):seen.append(ids);return .9,{},e.mask
        e.env['calculate_reward']=score
        row=dict(stopped=False,stop_reason='token_limit',text=text)
        if reference:row.update(gre=raw[:-1],raw_gre=raw[:-1])
        return e.evaluate(row),seen
    gre,a=evaluate(True);latex,b=evaluate(False)
    assert gre['gre']==latex['gre'] and a==b
    assert gre['status']==latex['status']=='evaluated'
    assert gre['reward']==latex['reward']==.9
    assert gre['normalized'] and latex['normalized']


def test_scoring_keeps_autograd_for_pde_derivatives():
    class DerivativeEvaluator(Recorder):
        def evaluate(self,row):
            assert torch.is_grad_enabled()
            x=torch.tensor(2.,requires_grad=True)
            assert torch.autograd.grad(x*x,x)[0].item()==4.
            return super().evaluate(row)
    sample_and_evaluate(Core(),Adapter(),DerivativeEvaluator(),2,2,525,False)


def test_incomplete_math_rejected_for_both_representations():
    from math_tokenizer.eqgpt.author import DICTIONARY
    w=DICTIONARY['word2id']
    for row in [dict(gre=[5,w['ut'],2],text='(u_t)+'),dict(text='(u_t)+')]:
        e=RewardEvaluator(None,None,'Wave_equation',['x','t'],'cpu',.2)
        e.env['calculate_reward']=lambda *args: (_ for _ in ()).throw(AssertionError('Must not score'))
        r=e.evaluate(dict(row,stopped=False,stop_reason='token_limit'))
        assert r['status']=='syntax_error'


def test_same_continuation_capacity_gate_for_both_representations():
    import pytest
    from math_tokenizer.eqgpt.paired_pipeline import continuation_sequences
    class Encoder:
        def sequence(self,value):return list(range(7))
    for reference in (False,True):
        with pytest.raises(ValueError,match='positional capacity'):
            continuation_sequences(Encoder(),[{'gre':[5,6,1]}],reference,5)
        assert len(continuation_sequences(Encoder(),[{'gre':[5,6,1]}],reference,6)[0])==7


def test_empty_and_interior_control_gre_rejected():
    for gre in ([5],[5,1],[5,6,1,3,7,1],[5,6,2,1]):
        e=RewardEvaluator(None,None,'Wave_equation',['x','t'],'cpu',.2)
        row=e.evaluate(dict(gre=gre,text='u_t',stopped=False,stop_reason='token_limit'))
        assert row['status']=='syntax_error'
