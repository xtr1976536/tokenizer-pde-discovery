import gzip, json, time, platform, subprocess
from pathlib import Path
import torch
from scripts.pde_experiment import verify
from math_tokenizer.corpus import ROOT
from math_tokenizer.eqgpt.author import verify_source, training_namespace
from math_tokenizer.eqgpt.models import build_models
from math_tokenizer.eqgpt.tokenizer_adapters import Adapter
from math_tokenizer.eqgpt.compact_alphabet import CompactAlphabet
from math_tokenizer.eqgpt.latex_tokenizers import LatexAdapter
from math_tokenizer.eqgpt.existing_surrogate import load_existing
from math_tokenizer.eqgpt.shared8_config import SHARED8_CASES

assert torch.cuda.is_available(), 'CUDA is required for this run'
torch.set_num_threads(8)
folder, manifest = verify(636)
verify_source()
with gzip.open(folder/'paired.jsonl.gz','rt') as stream:
    rows = [json.loads(line) for line in stream]
assert len(rows) == 636*32
adapters = {'eqgpt_reference': Adapter('eqgpt')}
for n in (280,315,350):
    adapters[f'ast_bpe_{n}'] = LatexAdapter('ast_bpe', CompactAlphabet.load(folder/'tokenizers'/f'ast_bpe_{n}.model.json'))
sequence_report = {}
for arm, adapter in adapters.items():
    sequences = [adapter.sequence(r['gre_ids'] if arm=='eqgpt_reference' else r['latex']) for r in rows]
    assert max(map(len,sequences)) <= 50
    sequence_report[arm] = dict(samples=len(sequences),max_tokens=max(map(len,sequences)))
    print('CORPUS_OK',arm,sequence_report[arm],flush=True)

# Check worst-length batch memory with the registered architecture and batch128.
# This is an isolated forward/backward validation, without an optimizer update.
models, metadata = build_models({'eqgpt_reference':adapters['eqgpt_reference']},50,525,'cuda',n_layers=6)
model = models['eqgpt_reference']
sequences = sorted((r['gre_ids'] for r in rows),key=len,reverse=True)[:128]
env = training_namespace('cuda'); ds = env['MyDataSet'](sequences)
x,y = ds.padding_batch([ds[i] for i in range(len(ds))]);x,y=x.cuda(),y.cuda()
torch.cuda.reset_peak_memory_stats();started=time.perf_counter()
logits,_ = model(x);loss=torch.nn.functional.cross_entropy(logits,y.flatten(),ignore_index=0)
loss.backward();torch.cuda.synchronize()
memory = dict(batch=128,sequence_length=int(x.shape[1]),peak_allocated_bytes=torch.cuda.max_memory_allocated(),
              seconds=time.perf_counter()-started,loss=float(loss.detach()))
assert torch.isfinite(loss)
print('GPU_BATCH_OK',memory,flush=True)
del model,models,x,y,logits,loss
import gc
gc.collect();torch.cuda.empty_cache()
cache_reports = {}
for case in SHARED8_CASES:
    _,net,database,_,report=load_existing(ROOT/'data/mlp_cache'/case,case,'cuda',525,10000)
    cache_reports[case] = report['cache_validation']
    del net,database;gc.collect();torch.cuda.empty_cache()
    print('MLP_CACHE_OK',case,flush=True)
report = dict(status='PASS',scale=636,sources=636,augmented_samples=len(rows),python=platform.python_version(),
              torch=str(torch.__version__),cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(),
              commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
              corpus=sequence_report,gpu_batch_validation=memory,mlp_caches=cache_reports)
out=ROOT.parent.parent/'outputs/preflight636.json';out.parent.mkdir(parents=True,exist_ok=True)
out.write_text(json.dumps(report,indent=2)+'\n')
