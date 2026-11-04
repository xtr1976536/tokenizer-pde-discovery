"""Shared exact-byte command/number counts and token-use metrics from the legacy study."""
import bisect,hashlib,re
from collections import Counter,defaultdict
from math_tokenizer.metrics_v2 import token_bytes
LEX=re.compile(r'\\[A-Za-z]+|\\[^A-Za-z]|[0-9]+(?:\.[0-9]+)?|[\s\S]')

def sha(value):
    return hashlib.sha256(value if isinstance(value,bytes) else value.encode()).hexdigest()

def measure(model,rows,own_initial=None):
    rr=[];counts=Counter()
    for row in rows:
        text=row['latex'];raw=text.encode();ids=model.encode(text)
        pieces=[model.vocab[i] for i in ids] if hasattr(model,'vocab') else [token_bytes(model,int(i)) for i in ids]
        assert all(pieces) and b''.join(pieces)==raw and model.decode(ids)==text
        counts.update(ids)
        bounds=[];pos=0
        for b in pieces[:-1]:
            pos+=len(b);bounds.append(pos)
        cmd=cmdok=num=numok=0
        for m in LEX.finditer(text):
            token=m.group();a=len(text[:m.start()].encode());b=a+len(token.encode())
            intact=bisect.bisect_right(bounds,a)==bisect.bisect_left(bounds,b)
            if re.fullmatch(r'\\[A-Za-z]+',token):
                cmd+=1;cmdok+=intact
            if re.fullmatch(r'[0-9]+(?:\.[0-9]+)?',token):
                num+=1;numok+=intact
        rr.append(dict(id=row['id'],source_group=row['source_group'],complexity=row['complexity_bin'],
            initial_character_tokens=len(raw),initial_method_tokens=len(own_initial.symbols(text)) if own_initial else len(raw),
            final_tokens=len(ids),commands=cmd,commands_intact=cmdok,numbers=num,numbers_intact=numok))
    total=Counter()
    for r in rr:
        total.update({k:v for k,v in r.items() if isinstance(v,int)})
    summary=dict(formulas=len(rr),sources=len({r['source_group'] for r in rr}),
        compression_ratio=total['initial_character_tokens']/total['final_tokens'],
        method_initial_compression_ratio=total['initial_method_tokens']/total['final_tokens'],
        cmd_pct=100*total['commands_intact']/total['commands'] if total['commands'] else None,
        num_pct=100*total['numbers_intact']/total['numbers'] if total['numbers'] else None,
        counts=dict(total),test_ids_sha256=sha('\n'.join(r['id'] for r in rows)))
    return summary,rr,counts

def token_coverage(model,rows):
    counts=Counter();formulas=Counter();papers=defaultdict(set)
    for row in rows:
        ids=model.encode(row['latex'])
        counts.update(ids)
        for i in set(ids):
            formulas[i]+=1;papers[i].add(row['source_group'])
    return counts,formulas,{i:len(v) for i,v in papers.items()}
