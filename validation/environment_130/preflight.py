import os,json,hashlib,subprocess,sys,time,gc
from pathlib import Path
import torch
ROOT=Path('/data1/users/liugang/src/eq-bpe/project');os.chdir(ROOT)
OUT=ROOT/'validation/environment_130';OUT.mkdir(parents=True,exist_ok=True)
def snapshot():
 return {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for name in ['data','eqgpt_original','experiments'] for p in (ROOT/name).rglob('*') if p.is_file()}
def run(args,log):
 with (OUT/log).open('w') as f:subprocess.run([sys.executable,*args],stdout=f,stderr=subprocess.STDOUT,check=True)
start=time.time();before=snapshot();(OUT/'asset_hashes_before.json').write_text(json.dumps(before,indent=2))
assert torch.cuda.is_available() and torch.cuda.device_count()==1
print('CUDA',torch.__version__,torch.version.cuda,torch.cuda.get_device_name(0),flush=True)
torch.set_num_threads(8)
a=torch.randn(128,128,device='cuda',requires_grad=True);a.square().mean().backward();torch.cuda.synchronize();assert torch.isfinite(a.grad).all();del a
run(['-m','pytest','-q'],'pytest.log');print('PYTEST PASS',flush=True)
run([str(OUT/'audit_inputs.py'),'--device','cuda'],'input_audit.log');print('ALL INPUTS/8 MODEL BACKWARDS PASS',flush=True)
from math_tokenizer.eqgpt.existing_surrogate import load_existing
from math_tokenizer.eqgpt.shared8_config import SHARED8_CASES
cache_checks={}
for case in SHARED8_CASES:
 env,net,db,state,report=load_existing(ROOT/'data/mlp_cache'/case,case,'cuda',525,10000)
 cache_checks[case]=report['cache_validation'];del env,net,db,state;gc.collect();torch.cuda.empty_cache()
 print('CACHE PASS',case,flush=True)
(OUT/'surrogate_replays.json').write_text(json.dumps(cache_checks,indent=2))
from scripts.pde_experiment import effective_settings
expected=effective_settings(130)
for scale in [212,636]:
 plan=json.loads(subprocess.check_output([sys.executable,'-m','scripts.pde_experiment','plan','--scale',str(scale),'--epochs','130']))
 assert plan['settings']==expected
 dest=OUT/f'smoke_{scale}'
 run(['-u','-m','scripts.pde_experiment','smoke','--scale',str(scale),'--device','cuda','--epochs','130','--output',str(dest)],f'smoke_{scale}.log')
 complete=json.loads((dest/'COMPLETE.json').read_text());assert complete['status']=='PASS' and len(complete['results'])==4
 print('SMOKE PASS',scale,flush=True)
after=snapshot();assert before==after,'Protected historical assets changed'
report=dict(status='READY',formal_training_started=False,torch=torch.__version__,cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(0),settings=expected,scales=[212,636],pytest=True,all_training_inputs_checked=True,backward_checks=8,surrogate_cache_replays=len(cache_checks),smoke_scales=[212,636],protected_asset_files=len(before),protected_assets_unchanged=True,seconds=time.time()-start)
(OUT/'READY.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
