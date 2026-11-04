"""Compact, read-only progress view for the full636 supervisor."""
import datetime,json,re,time
from pathlib import Path
root=Path(__file__).resolve().parent.parent
out=root/'outputs';last=None
while True:
    process=json.loads((out/'process636.json').read_text())
    with (out/'full636.log').open('rb') as stream:
        stream.seek(max(0,(out/'full636.log').stat().st_size-128*1024))
        log=stream.read().replace(b'\0',b'').decode('utf-8',errors='replace')
    relevant=[line.strip() for line in log.splitlines() if re.match(r'^(ARM_START|CASE_START|pretrain |CANDIDATE_CHECKPOINT|STUDY3_CYCLE|SEARCH_RESUME|CANDIDATE_RESUME)',line)]
    point=relevant[-1] if relevant else 'initializing runtime'
    state=dict(status=process['status'],point=point,
               trained_models=len(list((out/'large636/shared_pretrain').glob('*/training.json'))),
               completed_case_models=len(list((out/'large636/cases').glob('*/*/complete.json'))))
    key=json.dumps(state,sort_keys=True)
    if key!=last:
        state['updated_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
        target=out/'progress636.json';temp=target.with_suffix('.json.part')
        temp.write_text(json.dumps(state,indent=2)+'\n');temp.replace(target)
        print(json.dumps(state,ensure_ascii=False),flush=True);last=key
    if process['status']!='running':
        if process['status']=='failed':print(log[-12000:],flush=True)
        break
    time.sleep(5)
