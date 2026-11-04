"""Package the audited run, preserving the large final weights in their folder."""
import hashlib,json,subprocess,zipfile
from pathlib import Path
root=Path(__file__).resolve().parent.parent
out=root/'outputs';repo=root/'work/tokenizer-pde-discovery'
summary=json.loads((out/'summary636.json').read_text())
assert summary['status']=='PASS' and summary['artifact_hashes_verified']
assert (out/'large636/COMPLETE.json').exists()

# WSL emits its startup warning in UTF-16 while Python emits UTF-8.
raw=(out/'full636.log').read_bytes();parts=[];position=0
while True:
    start=raw.find(b'w\0s\0l\0:\0',position)
    if start<0:
        parts.append(raw[position:].decode('utf-8',errors='replace'));break
    parts.append(raw[position:start].decode('utf-8',errors='replace'))
    end=raw.find(b'\r\0\n\0',start)
    if end<0:
        parts.append(raw[start:].replace(b'\0',b'').decode('utf-8',errors='replace'));break
    end+=4
    parts.append(raw[start:end].decode('utf-16le',errors='replace'))
    position=end
(out/'experiment636.log').write_text(''.join(parts),encoding='utf-8')

files=subprocess.check_output(['git','ls-files','-z'],cwd=repo).decode().split('\0')
files += ['src/math_tokenizer/eqgpt/checkpointing.py','tests/test_restart.py']
helpers=['runtime_env.sh','run_full636.sh','guard_full636.py','retain636.py',
         'preflight636.py','check_runtime.sh','watch636.py','finalize636.py',
         'verify_recovery636.py','check_recovery636.sh','copy_storage636.py',
         'storage636_atomic_probe.py','check_storage636.sh','package636.py',
         'record_recovery98_636.py','await_recovery98_636.py',
         'record_search_recovery636.py','verify_search_recovery636.py','check_search_recovery636.sh',
         'verify_compression636.py']
evidence=['summary636.json','summary636.txt','preflight636.json','verification.txt',
          'runtime-requirements.freeze.txt','storage_policy636.json','experiment636.log']
for optional in ('restarts636.json','recovery636-stack.txt','storage_migration636.json',
                 'storage_migration636_hashes.json','recovery636-stack-ast315-epoch98.txt',
                 'recovery98-validation.log','recovery636-stack-ast350-search.txt',
                 'recovery-search636-validation.json','cleanup636.json','compression636.json'):
    if (out/optional).exists():evidence.append(optional)
temporary=out/'runtime-source.zip.part'
with zipfile.ZipFile(temporary,'w',zipfile.ZIP_DEFLATED) as archive:
    for name in files:
        if name:archive.write(repo/name,'tokenizer-pde-discovery/'+name)
    for name in helpers:archive.write(root/'work'/name,'launch/'+name)
    archive.write(root/'.gitignore','launch/workspace.gitignore')
    archive.write(out/'resume636.ps1','launch/resume636.ps1')
    for name in evidence:archive.write(out/name,'run-evidence/'+name)
    archive.write(out/'large636/RUN.json','run-evidence/RUN.json')
    archive.write(out/'large636/COMPLETE.json','run-evidence/COMPLETE.json')
    archive.writestr('READ_ME_FIRST.txt',
        '本包包含已完成的636正式实验所用源码、启动与校验工具、当前运行证据。\n'
        '上游README及experiments/smoke记录的是此前发布历史；本次正式结果见run-evidence。\n'
        '仅执行636组，未执行212组或smoke实验。\n'
        '最终模型权重和完整候选轨迹单独保存在outputs/large636，避免在压缩包重复占用空间。\n'
        '启动工具保留本次工作站路径；运行环境版本见runtime-requirements.freeze.txt。\n')
with zipfile.ZipFile(temporary) as archive:assert archive.testzip() is None
temporary.replace(out/'runtime-source.zip')
manifest={}
for name in evidence+['runtime-source.zip']:
    path=out/name
    with path.open('rb') as stream:manifest[name]=hashlib.file_digest(stream,'sha256').hexdigest()
(out/'deliverables636.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
print('Audited source and result package saved:',out/'runtime-source.zip')
