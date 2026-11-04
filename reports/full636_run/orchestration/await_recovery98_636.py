"""Verify resumed Adam state as soon as epoch99 is atomically committed."""
import subprocess,time,sys
from pathlib import Path
root=Path(__file__).resolve().parent.parent
sys.stdout.reconfigure(encoding='utf-8',errors='replace')
dest=root/'outputs/large636'
source=dest/'shared_pretrain/ast_bpe_315/training_progress.pt'
backup=dest/'recovery636-ast315-epoch98.pt'
deadline=time.monotonic()+1800
while time.monotonic()<deadline:
    if source.exists() and source.stat().st_ino!=backup.stat().st_ino:
        result=subprocess.run(['wsl','-d','Ubuntu-20.04','--','bash',
            '/mnt/c/Users/28678/Documents/Codex/2026-11-04/https-github-com-xtr1976536-tokenizer-pde/work/check_recovery636.sh'],
            stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
        start=result.stdout.find(b'{\n')
        output=result.stdout[max(start,0):].decode('utf-8',errors='replace')
        (root/'outputs/recovery98-validation.log').write_text(output,encoding='utf-8')
        print(output,flush=True)
        raise SystemExit(result.returncode)
    time.sleep(2)
raise SystemExit('No resumed epoch checkpoint within 30 minutes')
