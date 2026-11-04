"""Preserve the committed epoch98 state and evidence before a scoped restart."""
import datetime,hashlib,json,os,subprocess
from pathlib import Path
root=Path(__file__).resolve().parent.parent
out=root/'outputs';dest=out/'large636'
source=dest/'shared_pretrain/ast_bpe_315/training_progress.pt'
backup=dest/'recovery636-ast315-epoch98.pt'
assert source.stat().st_size==384137093
assert not backup.exists()
command=['wsl','-d','Ubuntu-20.04','-u','root','--',
    '/mnt/c/Users/28678/Documents/Codex/2026-11-04/https-github-com-xtr1976536-tokenizer-pde/work/linux-probe/bin/py-spy',
    'dump','--pid','931']
stack=subprocess.check_output(command)
assert b'Process 931:' in stack and b'get_attn_subsequence_mask (gpt_model.py:82)' in stack
stack=stack[stack.index(b'Process 931:'):].decode('utf-8',errors='replace')
(out/'recovery636-stack-ast315-epoch98.txt').write_text(stack,encoding='utf-8')
os.link(source,backup)
with backup.open('rb') as stream:digest=hashlib.file_digest(stream,'sha256').hexdigest()
path=out/'restarts636.json';records=json.loads(path.read_text())
records.append(dict(incident='cuda_host_to_device_stall',stage='ast_bpe_315_pretraining',
    arm='ast_bpe_315',completed_epoch_checkpoint=98,
    checkpoint_backup_relative=backup.relative_to(root).as_posix(),checkpoint_sha256=digest,
    mathematical_source_changed=False,completed_case_models_reused=16,
    stack_file='recovery636-stack-ast315-epoch98.txt',
    utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))
temporary=path.with_suffix('.json.part');temporary.write_text(json.dumps(records,indent=2)+'\n');temporary.replace(path)
print(json.dumps(records[-1],indent=2))
