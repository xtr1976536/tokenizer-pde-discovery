"""Matched downstream protocol: representation and EqGPT-only SAT may differ."""
import hashlib
import json
from pathlib import Path
import torch
from .author import seed_all
from .latex_codec import structure_to_latex

VERSION = 'paired_pipeline_v3'
POLICY = dict(version=VERSION, sampling='80% categorical +20% uniform over permitted IDs',
              sat='EqGPT only; Hard has no grammar/physical/control-ID generation mask',
              candidate_order='serial generation then immediate shared evaluation',
              randomness='seed + cycle*attempts + attempt; independent candidate streams',
              length='maximum NEW predictions after BOS, including EOS',
              terminal='EOS or token_limit; exact representation must be parseable',
              cleanup='shared author delete_duplicate, including its documented operator-blind behavior',
              failures='shared status filtering; non-evaluated candidates cannot enter Top10',
              selection='shared float32 reward dedup, same merge and available-candidate policy',
              continuation='same optimizer/model binding and cleaned structure inputs')


def fingerprint():
    # Bind the dependency family, not only the launcher, before reusing results.
    root=Path(__file__).resolve().parents[3]
    files=sorted(set(root.glob('src/**/*.py')) |
                 set(root.glob('eqgpt_original/code/*.py')))
    payload={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    return hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest()


def sample_and_evaluate(model,adapter,evaluator,attempts,limit,seed,reference=False):
    if attempts<1 or limit<1:raise ValueError('Invalid sampling budget')
    core=getattr(model,'core',model)
    if limit>core.decoder.pos_emb.num_embeddings:raise ValueError('Generation exceeds positional capacity')
    model.eval();device=next(core.parameters()).device
    rows=[]
    for attempt in range(attempts):
        # Python/NumPy may be used in scoring; freeze their candidate-level streams too.
        seed_all(seed+attempt)
        rng=torch.Generator(device=device).manual_seed(seed+attempt)
        prefix=[adapter.bos];payload=[];reason='token_limit'
        for _ in range(limit):
            inputs=torch.tensor([prefix],dtype=torch.long,device=device)
            with torch.no_grad():
                logits,_=core(inputs)
            probs=logits.reshape(1,len(prefix),adapter.vocab_size)[0,-1].softmax(-1)
            allowed=torch.ones(adapter.vocab_size,dtype=torch.bool,device=device)
            if reference:
                allowed=evaluator.mask.to(device)!=0
                if len(prefix)%2==0:
                    allowed[0]=False;allowed[5:]=False
                else:allowed[:6]=False
            probs=probs*allowed
            mass=probs.sum()
            if not torch.isfinite(mass) or mass<=0:raise FloatingPointError('No finite permitted probability mass')
            probs=probs/mass
            if torch.rand((),device=device,generator=rng).item()<=.8:
                token=int(torch.multinomial(probs,1,generator=rng))
            else:
                valid=torch.nonzero(probs!=0).flatten()
                token=int(valid[torch.randint(len(valid),(1,),device=device,generator=rng)].item())
            payload.append(token);prefix.append(token)
            if token==adapter.eos:reason='eos';break
            if token in (adapter.pad,adapter.bos):reason='unexpected_control';break
        row=dict(attempt=attempt,reference=reference,ids=prefix if reference else payload,
                 stopped=reason=='eos',stop_reason=reason,sampling_policy=VERSION,
                 generated_tokens=len(payload),sampling_seed=seed+attempt)
        if reference:
            row['raw_gre']=prefix.copy();row['gre']=prefix.copy()
            row['text']=structure_to_latex(prefix)
        else:
            try:row['text']=adapter.decode(payload[:-1] if reason=='eos' else payload)
            except (UnicodeDecodeError,ValueError,IndexError):row.update(text=None,decode_error=True)
        rows.append(evaluator.evaluate(row))
    return rows


def continuation_sequences(adapter,best,reference,capacity):
    values=[r['gre'] if reference else structure_to_latex(r['gre']) for r in best]
    sequences=[adapter.sequence(value) for value in values]
    if any(len(seq)-1>capacity for seq in sequences):
        raise ValueError('Continuation input exceeds positional capacity')
    return sequences


def runtime_bindings(cohorts,cases):
    from .author import ROOT
    import platform
    import numpy
    def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
    files={}
    for cohort in cohorts:
        folder=ROOT/'experiments/study3_corpus_effect_v1'/cohort/'shared_pretrain'
        for path in sorted(folder.glob('*/*')):
            if path.name in ('pretrained.pt','training.json'):
                files[str(path.relative_to(ROOT))]=digest(path)
    for case in cases:
        for name in ('surrogate_cache.pt','surrogate_quality.json'):
            path=ROOT/'data/mlp_cache'/case/name
            files[str(path.relative_to(ROOT))]=digest(path)
    return dict(files=files,python=platform.python_version(),torch=torch.__version__,numpy=numpy.__version__,
                machine=platform.machine(),platform=platform.platform())
