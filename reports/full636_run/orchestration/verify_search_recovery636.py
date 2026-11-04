"""Verify preserved search model, Adam, trace, and candidate prefix on recovery."""
import datetime,hashlib,json,time
from pathlib import Path
import torch
root=Path(__file__).resolve().parent.parent
path=root/'outputs/restarts636.json';records=json.loads(path.read_text())
index=next(i for i in reversed(range(len(records))) if records[i].get('kind')=='search' and not records[i].get('recovery_verified'))
entry=records[index];backup=root/entry['checkpoint_backup_relative']
folder=root/'outputs/large636/cases'/entry['case']/entry['arm']
for name,expected in entry['checkpoint_hashes'].items():
    with (backup/name).open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==expected
old=torch.load(backup/'search_progress.pt',map_location='cpu',weights_only=True)
old_attempts=torch.load(backup/'attempt_progress.pt',map_location='cpu',weights_only=True)
assert old['cycle']==old_attempts['cycle']==entry['cycle']-1
assert len(old_attempts['evaluated'])==entry['completed_attempts_in_cycle']
deadline=time.monotonic()+1800
while time.monotonic()<deadline:
    current_path=folder/'attempt_progress.pt'
    if current_path.exists() and current_path.stat().st_ino!=(backup/'attempt_progress.pt').stat().st_ino:break
    time.sleep(1)
else:raise TimeoutError('No resumed candidate checkpoint within 30 minutes')
current=torch.load(folder/'search_progress.pt',map_location='cpu',weights_only=True)
attempts=torch.load(current_path,map_location='cpu',weights_only=True)
assert current['identity']==old['identity']==attempts['identity']==old_attempts['identity']
assert current['cycle']==old['cycle'] and attempts['cycle']==old_attempts['cycle']
assert current['history']==old['history'] and current['trace']==old['trace']
assert attempts['evaluated'][:len(old_attempts['evaluated'])]==old_attempts['evaluated']
assert len(attempts['evaluated'])>len(old_attempts['evaluated'])
assert all(torch.equal(v,current['model'][k]) for k,v in old['model'].items())
assert current['optimizer']['param_groups']==old['optimizer']['param_groups']
for key,state in old['optimizer']['state'].items():
    for name,value in state.items():
        other=current['optimizer']['state'][key][name]
        assert torch.equal(value,other) if isinstance(value,torch.Tensor) else value==other
entry.update(recovery_verified=True,model_and_optimizer_preserved=True,
    completed_cycle_history_preserved=True,candidate_prefix_preserved=True,
    rng_and_evaluator_state_retained=True,resumed_attempt_checkpoint=len(attempts['evaluated']),
    optimizer_steps_preserved=sorted({int(s['step'].item()) for s in old['optimizer']['state'].values()}),
    verified_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
# Re-read records to preserve any independent evidence appended while waiting.
records=json.loads(path.read_text());records[index]=entry
temp=path.with_suffix('.json.part');temp.write_text(json.dumps(records,indent=2)+'\n');temp.replace(path)
text=json.dumps(entry,indent=2)+'\n'
(root/'outputs/recovery-search636-validation.json').write_text(text)
print(text,flush=True)
