"""Release only completed diagnostic-only cycle weights; retain restart/final states."""
import datetime, hashlib, json, time, msvcrt
from pathlib import Path
root=Path(__file__).resolve().parent.parent
lock=open(root/'work/retention636.lock','a+b')
lock.seek(0);lock.write(b'0');lock.flush();lock.seek(0)
try:msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
except OSError:raise SystemExit('A storage retention worker is already running')
out=root/'outputs';dest=out/'large636'
policy=dict(policy='retain_pretrain_and_final_weights_v1',
            retained=['shared_pretrain/*/pretrained.pt','cases/*/*/cycle_0.pt',
                      'cases/*/*/cycle_5.pt','cases/*/*/searched.pt',
                      'all active training/search/candidate restart states',
                      'all candidate traces, histories, diagnostics and weight hashes'],
            released='cycle_1.pt through cycle_4.pt only AFTER diagnostics and case completion',
            reason='bounded artifact storage; mathematical runtime and budgets remain unchanged')

def dump(path,value):
    temp=path.with_suffix(path.suffix+'.part')
    temp.write_text(json.dumps(value,indent=2)+'\n');temp.replace(path)

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()

dump(out/'storage_policy636.json',policy)
while True:
    for marker in (dest/'cases').glob('*/*/complete.json'):
        folder=marker.parent
        manifest=json.loads(marker.read_text())
        retention=folder/'retention.json'
        removed=json.loads(retention.read_text())['released_files'] if retention.exists() else {}
        names=[f'cycle_{i}.pt' for i in range(1,5)]
        changed=False
        for name in names:
            if name in manifest['files']:
                assert sha(folder/name)==manifest['files'][name],str(folder/name)
                removed[name]=manifest['files'].pop(name);changed=True
        if changed:
            dump(retention,dict(policy=policy['policy'],released_files=removed,
                 released_after_diagnostics=True,
                 updated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))
            manifest['files']['retention.json']=sha(retention)
            dump(marker,manifest)
        # A committed marker no longer depends on these files, even after a crash.
        for name,expected in removed.items():
            path=folder/name
            if path.exists():
                assert sha(path)==expected,str(path)
                path.unlink()
                print('RELEASED',str(path.relative_to(dest)),flush=True)
    process=json.loads((out/'process636.json').read_text())
    if process['status']!='running':break
    time.sleep(10)
