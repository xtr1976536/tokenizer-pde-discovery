"""Snapshot the two committed search restart files before a scoped GPU restart."""
import datetime,hashlib,json,os,subprocess
from pathlib import Path
root=Path(__file__).resolve().parent.parent
out=root/'outputs';dest=out/'large636'
folder=dest/'cases/Convection_diffusion_equation/ast_bpe_350'
backup=dest/'recovery-search636-ast350-convection'
backup.mkdir(exist_ok=False)
stack=subprocess.check_output(['wsl','-d','Ubuntu-20.04','-u','root','--',
    '/mnt/c/Users/28678/Documents/Codex/2026-11-04/https-github-com-xtr1976536-tokenizer-pde/work/linux-probe/bin/py-spy',
    'dump','--pid','678'])
assert b'Process 678:' in stack and b'paired_pipeline.py:56' in stack
(out/'recovery636-stack-ast350-search.txt').write_text(
    stack[stack.index(b'Process 678:'):].decode('utf-8',errors='replace'),encoding='utf-8')
hashes={}
for name in ('search_progress.pt','attempt_progress.pt'):
    os.link(folder/name,backup/name)
    with (backup/name).open('rb') as f:hashes[name]=hashlib.file_digest(f,'sha256').hexdigest()
path=out/'restarts636.json';records=json.loads(path.read_text())
records.append(dict(incident='cuda_synchronization_stall',kind='search',
    stage='ast_bpe_350_Convection_diffusion_equation_search',arm='ast_bpe_350',
    case='Convection_diffusion_equation',cycle=2,completed_attempts_in_cycle=75,
    checkpoint_backup_relative=backup.relative_to(root).as_posix(),checkpoint_hashes=hashes,
    mathematical_source_changed=False,completed_case_models_reused=25,
    stack_file='recovery636-stack-ast350-search.txt',
    utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))
temp=path.with_suffix('.json.part');temp.write_text(json.dumps(records,indent=2)+'\n');temp.replace(path)
print(json.dumps(records[-1],indent=2))
