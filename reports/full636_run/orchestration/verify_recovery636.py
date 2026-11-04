import datetime,hashlib,json
from pathlib import Path
import torch
root=Path(__file__).resolve().parent.parent
path=root/'outputs/restarts636.json';record=json.loads(path.read_text())
pending=[i for i,r in enumerate(record) if not r.get('recovery_verified')]
if not pending:
    print('All recorded recoveries are already verified');raise SystemExit(0)
index=pending[-1];entry=record[index]
arm=entry.get('arm',entry['stage'].removesuffix('_pretraining'))
epoch=entry['completed_epoch_checkpoint']
old_path=root/entry.get('checkpoint_backup_relative','work/recovery636-ast280-epoch31.pt')
current_path=root/'outputs/large636/shared_pretrain'/arm/'training_progress.pt'
old=torch.load(old_path,map_location='cpu',weights_only=True)
current=torch.load(current_path,map_location='cpu',weights_only=True)
assert old['epoch']==epoch and current['epoch']>epoch
for key in ('identity','initial','epochs','batch'):assert old[key]==current[key],key
assert current['history'][:epoch]==old['history']
old_steps={int(s['step'].item()) for s in old['optimizer']['state'].values()}
current_steps={int(s['step'].item()) for s in current['optimizer']['state'].values()}
assert old_steps=={epoch*159} and current_steps=={current['epoch']*159}
changed=sum(not torch.equal(old['model'][k],v) for k,v in current['model'].items())
assert changed>0
assert hashlib.file_digest(old_path.open('rb'),'sha256').hexdigest()==entry['checkpoint_sha256']
entry.update(resumed_epoch=current['epoch'],history_prefix_preserved=True,
                 optimizer_steps_before=epoch*159,optimizer_steps_after=current['epoch']*159,
                 changed_parameter_tensors=changed,recovery_verified=True,
                 verified_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
temp=path.with_suffix('.json.part');temp.write_text(json.dumps(record,indent=2)+'\n');temp.replace(path)
print(json.dumps(entry,indent=2),flush=True)
