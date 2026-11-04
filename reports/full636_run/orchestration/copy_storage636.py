"""Copy this stopped run to its dedicated D: backing folder, preserving hardlinks."""
import datetime,hashlib,json,os,shutil,stat
from pathlib import Path
root=Path(__file__).resolve().parent.parent
source=root/'outputs/large636'
backing=Path('D:/desktop/CodexRuns/2026-11-04-tokenizer-pde-636-01a1a261')
target=backing/'large636'
assert source.resolve()==root/'outputs/large636' and not source.is_junction()
assert backing.resolve()==backing
assert json.loads((backing/'task.json').read_text())['workspace']==str(root)
assert json.loads((root/'outputs/process636.json').read_text())['status']!='running'
assert not target.exists(),str(target)
files=[];skipped=[];unique={}
for path in sorted(source.rglob('*')):
    assert not path.is_symlink() and not path.is_junction(),str(path)
    if path.is_dir():continue
    info=path.stat();assert stat.S_ISREG(info.st_mode) and info.st_ino!=0
    if path.name.endswith('.part'):
        skipped.append(path.relative_to(source).as_posix());continue
    key=(info.st_dev,info.st_ino)
    unique.setdefault(key,info.st_size)
    files.append((path,key))
unique_bytes=sum(unique.values())
assert shutil.disk_usage(backing).free>unique_bytes+1024**3
target.mkdir()
copied={};records={};done=0
def digest(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
print(json.dumps(dict(files=len(files),unique_files=len(unique),unique_bytes=unique_bytes)),flush=True)
for path,key in files:
    name=path.relative_to(source).as_posix();destination=target/name
    destination.parent.mkdir(parents=True,exist_ok=True)
    if key in copied:
        previous,expected=copied[key];os.link(previous,destination)
        assert destination.samefile(previous)
    else:
        shutil.copy2(path,destination)
        expected=digest(path);assert digest(destination)==expected,name
        copied[key]=(destination,expected);done+=path.stat().st_size
        if path.suffix=='.pt':print('VERIFIED',name,done,flush=True)
    records[name]=expected
assert len(records)==len(files)
manifest=dict(source=str(source),target=str(target),files=records,unique_files=len(unique),
              unique_bytes=unique_bytes,all_unique_files_sha256_verified=True,
              hardlinks_preserved=True,skipped_incomplete_temporary_files=skipped,
              copied_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
path=root/'work/storage636-copy-manifest.json'
path.write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
print('COPY_AND_HASH_VERIFICATION_PASS',flush=True)
