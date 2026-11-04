"""Run only full636, keep Windows awake, and leave durable process evidence."""
import ctypes
import datetime
import json
import msvcrt
import os
from pathlib import Path
import subprocess
import sys
import time

root = Path(__file__).resolve().parent.parent
out = root/'outputs';out.mkdir(exist_ok=True)
lock = open(root/'work/full636.lock', 'a+b')
lock.seek(0);lock.write(b'0');lock.flush();lock.seek(0)
try:
    msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
except OSError:
    raise SystemExit('The full636 supervisor is already running')

def status(value):
    value.update(supervisor_pid=os.getpid(),updated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    target=out/'process636.json';temporary=target.with_suffix('.json.part')
    temporary.write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8');temporary.replace(target)

ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
try:
    with open(out/'full636.log','ab',buffering=0) as log:
        command=['wsl.exe','-d','Ubuntu-20.04','--','bash',
                 '/mnt/c/Users/28678/Documents/Codex/2026-11-04/https-github-com-xtr1976536-tokenizer-pde/work/run_full636.sh']
        process=subprocess.Popen(command,cwd=root,stdout=log,stderr=subprocess.STDOUT,
                                 creationflags=subprocess.CREATE_NO_WINDOW)
        status(dict(status='running',child_pid=process.pid,command=command,scale=636,smoke=False))
        # Storage cleanup touches only cases already committed complete.
        with open(out/'retention636.log','ab',buffering=0) as retention_log:
            subprocess.Popen([sys.executable,str(root/'work/retain636.py')],cwd=root,
                             stdout=retention_log,stderr=subprocess.STDOUT,
                             creationflags=subprocess.CREATE_NO_WINDOW)
        while process.poll() is None:
            time.sleep(30)
        code=process.returncode
        status(dict(status='completed' if code==0 and (out/'large636/COMPLETE.json').exists() else 'failed',
                    returncode=code,child_pid=process.pid,command=command,scale=636,smoke=False))
finally:
    ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
    lock.close()
