"""Confirm NTFS storage compression kept the retained recovery snapshots intact."""
import ctypes,hashlib,json
from pathlib import Path
root=Path(__file__).resolve().parent.parent
records=json.loads((root/'outputs/restarts636.json').read_text())
get_size=ctypes.windll.kernel32.GetCompressedFileSizeW
get_size.argtypes=[ctypes.c_wchar_p,ctypes.POINTER(ctypes.c_ulong)]
get_size.restype=ctypes.c_ulong
files={}
for entry in records:
    if entry.get('completed_epoch_checkpoint')==98:
        candidates={entry['checkpoint_backup_relative']:entry['checkpoint_sha256']}
    elif entry.get('kind')=='search':
        candidates={entry['checkpoint_backup_relative']+'/'+n:h for n,h in entry['checkpoint_hashes'].items()}
    else:continue
    for name,expected in candidates.items():
        path=root/name
        with path.open('rb') as f:actual=hashlib.file_digest(f,'sha256').hexdigest()
        assert actual==expected,name
        high=ctypes.c_ulong(0);low=get_size(str(path),ctypes.byref(high))
        files[name]=dict(logical_bytes=path.stat().st_size,stored_bytes=(high.value<<32)|low,
                        sha256=actual,content_unchanged=True,file_retained=True)
report=dict(method='NTFS transparent compression of immutable recovery snapshots',files=files,
            logical_bytes=sum(r['logical_bytes'] for r in files.values()),
            stored_bytes=sum(r['stored_bytes'] for r in files.values()),
            rejected_deletions_executed=False)
(root/'outputs/compression636.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items() if k!='files'},indent=2))
