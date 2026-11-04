"""Generate a small, verified trace from the repository's unchanged algorithms."""
from pathlib import Path
import sys, ctypes, json, heapq, importlib.util, os
from collections import Counter

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO / 'src'))
import math_tokenizer.frontend_v6 as frontend
import math_tokenizer.hard_ast_2026 as hard
from math_tokenizer.atoms_v4 import FastSpanSequence
from math_tokenizer.ast import compatible

# Only replace the platform build/load plumbing. Both C source files are unchanged.
if sys.platform == 'win32':
    native_dir = Path(os.environ.get('AST_BPE_NATIVE_DIR', REPO / '.native_cache'))
    lib = ctypes.CDLL(str(native_dir / 'frontend.dll'))
    for name in ('mt_lex', 'mt_tree'):
        getattr(lib, name).argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p]
        getattr(lib, name).restype = ctypes.c_int
    frontend._lib = lib
    spec = importlib.util.spec_from_file_location('_hard_ranked_2026', native_dir / '_hard_ranked_2026.pyd')
    backend = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(backend)
    hard._bridge = backend

corpus = ['u_t=u_x+u_{xx}', 'v_t=v_x+v_{xx}', 'u_t=u_x', 'v_t=v_x']
target = corpus[0]
model = hard.HardASTBPE.train(corpus, capacity=272)
def payload(i):
    return model.vocab[i].decode('utf-8', errors='replace')
def seq(text):
    raw = text.encode()
    return FastSpanSequence(list(raw), list(range(len(raw))), list(range(1,len(raw)+1)), frontend.structure_native(text).spans)
def snap(s):
    return [dict(id=i, a=a, b=b, text=payload(i)) for i,a,b in zip(s.ids,s.starts,s.ends)]
def pair_info(p, count):
    return dict(pair=list(p), text=[payload(i) for i in p], count=count)

seqs = [seq(t) for t in corpus]
training = []
for rank,(a,b,c) in enumerate(model.merges):
    counts = sum((s.counts() for s in seqs), Counter())
    best = min(counts, key=lambda p:(-counts[p],p))
    assert best == (a,b), (rank,best,(a,b))
    before = snap(seqs[0])
    eligible = [[s.starts[i],s.ends[i+1]] for s in seqs[:1] for i in range(len(s.ids)-1) if (s.ids[i],s.ids[i+1])==(a,b) and s.legal(i)]
    top = [pair_info(p,n) for p,n in sorted(counts.items(), key=lambda item:(-item[1],item[0]))[:4]]
    replacements = sum(s.gain((a,b)) for s in seqs)
    for s in seqs:s.merge((a,b),c)
    training.append(dict(rank=rank,rule=[a,b,c],new=payload(c),count=counts[(a,b)],replacements=replacements,top=top,before=before,after=snap(seqs[0]),ranges=eligible))

def trace_encode(text, m):
    raw=text.encode(); spans=frontend.structure_native(text).spans;n=len(raw)
    ids=list(raw); starts=list(range(n)); ends=list(range(1,n+1))
    following=list(range(1,n))+[-1] if n else [];previous=list(range(-1,n-1));version=[0]*n;heap=[]
    def offer(l,r):
        if l>=0 and r>=0 and (ids[l],ids[r]) in m.rules:
            rank,new=m.rules[(ids[l],ids[r])];heapq.heappush(heap,(rank,l,r,version[l],version[r],new))
    def state():
        return [dict(id=ids[i],a=starts[i],b=ends[i],text=m.vocab[ids[i]].decode(errors='replace')) for i in range(n) if following[i]!=-2]
    events=[]; initial=state()
    for i in range(n-1):offer(i,i+1)
    while heap:
        rank,l,r,vl,vr,new=heapq.heappop(heap)
        if following[l]!=r or version[l]!=vl or version[r]!=vr:continue
        before=state(); a=starts[l];b=ends[r];pair=[ids[l],ids[r]]
        blockers=[[x,y] for x,y in spans if not compatible(a,b,[(x,y)])]
        if blockers:
            events.append(dict(rank=rank,rule=[*pair,new],range=[a,b],blocked=blockers,before=before,after=before))
            continue
        ids[l]=new;ends[l]=ends[r];version[l]+=1;version[r]+=1;following[l]=following[r];following[r]=-2
        if following[l]>=0:previous[following[l]]=l
        offer(previous[l],l);offer(l,following[l])
        events.append(dict(rank=rank,rule=[*pair,new],range=[a,b],blocked=[],before=before,after=state()))
    result=[t['id'] for t in state()]
    assert result==m.encode(text)==m.encode_reference(text)
    assert m.decode(result)==text
    return dict(text=text,spans=spans,initial=initial,events=events,result=state())

encoded=trace_encode(target,model)
for t in corpus: trace_encode(t,model)
# Check against the persisted real 212-corpus model and every original source.
real=hard.HardASTBPE.load(REPO/'data/formulas/212/tokenizers/ast_bpe_280.model.backend.json')
sources=[json.loads(line)['original_latex'] for line in (REPO/'data/formulas/212/sources.jsonl').read_text(encoding='utf-8').splitlines()]
for t in sources: trace_encode(t,real)

# Show explicitly why a crossing pair is ineligible, using a real initial boundary.
initial=seq(target)
blocked=[]
for i in range(len(initial.ids)-1):
    if not initial.legal(i):
        a,b=initial.starts[i],initial.ends[i+1]
        blocked.append(dict(range=[a,b],text=target[a:b],spans=[[x,y] for x,y in initial.spans if not compatible(a,b,[(x,y)])]))

data=dict(commit='d1eda7638237dbec3ce4474e25d24d70820cab6e',corpus=corpus,target=target,capacity=len(model.vocab),training=training,encoding=encoded,blocked=blocked,vocab={str(i):payload(i) for i in range(len(model.vocab)) if i>=256 or i in target.encode()},verified=dict(native_reference_roundtrips=len(sources)+len(corpus)+1,training_rounds=len(training),real_model='data/formulas/212/tokenizers/ast_bpe_280.model.backend.json'))
(REPO/'visualization/trace.json').write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps(dict(verified=data['verified'],merges=[(e['rank'],e['new'],e['count']) for e in training],encoded=encoded['result'],blocked=blocked),ensure_ascii=False,indent=2))
