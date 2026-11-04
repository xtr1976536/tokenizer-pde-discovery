import datetime,json,os
from pathlib import Path
root=Path(__file__).resolve().parent.parent
assert json.loads((root/'outputs/process636.json').read_text())['status']!='running'
probe=root/'work/junction-probe636'
a=probe/'atomic-new.txt';b=probe/'atomic-old.txt'
a.write_text('replacement-confirmed\n');b.write_text('previous\n');a.replace(b)
assert b.read_text()=='replacement-confirmed\n'
run=root/'outputs/large636/RUN.json';part=run.with_suffix('.json.part')
if part.exists():
    assert part.read_bytes()==run.read_bytes()
    part.replace(run)
path=root/'outputs/storage_migration636.json';data=json.loads(path.read_text())
data.update(linux_atomic_replace_verified=True,
            atomic_replace_verified_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
temporary=path.with_suffix('.json.part');temporary.write_text(json.dumps(data,indent=2)+'\n');temporary.replace(path)
print('LINUX_ATOMIC_REPLACE_ON_D_JUNCTION_PASS',flush=True)
