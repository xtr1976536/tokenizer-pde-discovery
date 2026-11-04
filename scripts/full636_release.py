"""Upload/verify full636 weights without storing credentials or duplicate content."""
import argparse,concurrent.futures,hashlib,json,os,subprocess,time
from pathlib import Path
from urllib.parse import quote
import requests
ROOT=Path(__file__).resolve().parents[1]
REPO='xtr1976536/tokenizer-pde-discovery'
REPORT=ROOT/'reports/full636_run'
def auth():
 env=dict(os.environ,GIT_TERMINAL_PROMPT='0',GCM_INTERACTIVE='Never')
 p=subprocess.run(['git','credential','fill'],input='protocol=https\nhost=github.com\n\n',text=True,capture_output=True,env=env,check=True)
 d=dict(x.split('=',1) for x in p.stdout.splitlines() if '=' in x)
 if not d.get('password'):raise RuntimeError('GitHub Git credential unavailable')
 return {'Authorization':'Bearer '+d['password'],'Accept':'application/vnd.github+json','X-GitHub-Api-Version':'2022-11-28','User-Agent':'full636-archive'}
def digest(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
 return h.hexdigest()
def api(headers,path,method='GET',data=None):
 r=requests.request(method,'https://api.github.com/repos/'+REPO+path,headers=headers,json=data,timeout=(30,120));r.raise_for_status();return r.json()
def releases(headers):
 rows=[];page=1
 while True:
  batch=api(headers,f'/releases?per_page=100&page={page}');rows+=batch
  if len(batch)<100:return rows
  page+=1
def assets(headers,rid):
 rows=[];page=1
 while True:
  batch=api(headers,f'/releases/{rid}/assets?per_page=100&page={page}');rows+=batch
  if len(batch)<100:return {e['name']:e for e in rows}
  page+=1
def matches(a,e):return a and a['state']=='uploaded' and a['size']==e['bytes'] and a.get('digest')=='sha256:'+e['sha256']
def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('action',choices=['upload','verify','publish']);ap.add_argument('--run-root',type=Path);ap.add_argument('--local-map',type=Path);ap.add_argument('--workers',type=int,default=4);ap.add_argument('--previous-manifest-sha');a=ap.parse_args()
 m=json.loads((REPORT/'weights_manifest.json').read_text());h=auth();allreleases=releases(h);release=next((r for r in allreleases if r['tag_name']==m['tag']),None)
 if release is None:
  if a.action!='upload':raise RuntimeError('Release missing')
  release=api(h,'/releases','POST',dict(tag_name=m['tag'],target_commitish=m['training_base_commit'],name='Full 636-corpus experiment: checkpoints and recovery evidence',draft=True,prerelease=False,body='636 source formulas / 20,352 augmented sequences; four models, 32 case/model groups and 64,000 candidate attempts. Checkpoint contents are stored once and mapped to original paths by weights_manifest.json. Training base 815cc37 plus the archived runtime patch. See reports/full636_run for results, provenance, retention limits and three audit rounds.'))
 existing=assets(h,release['id']);new=[e for e in m['files'] if e['tag']==m['tag']]
 print('RELEASE',release['id'],'expected',len(new),'existing',len(existing),flush=True)
 if a.action=='upload':
  if a.run_root is None:ap.error('--run-root required for upload')
  local={e['sha256']:Path(e['path']) for e in json.loads(a.local_map.read_text())} if a.local_map else {}
  def upload(e):
   if matches(existing.get(e['asset']),e):return e['asset'],'already_verified'
   if e['asset'] in existing:raise RuntimeError('Existing asset mismatches: '+e['asset'])
   rel=Path(e['paths'][0]).relative_to('experiments/full/636');p=local.get(e['sha256'],a.run_root/rel)
   if p.stat().st_size!=e['bytes'] or digest(p)!=e['sha256']:raise RuntimeError('Local checkpoint mismatch: '+str(rel))
   for attempt in range(4):
    try:
     with p.open('rb') as f:
      r=requests.post(release['upload_url'].split('{')[0]+'?name='+quote(e['asset'],safe=''),headers={**h,'Content-Type':'application/octet-stream','Content-Length':str(e['bytes'])},data=f,timeout=(30,1800))
     if r.ok:
      if not matches(r.json(),e):raise RuntimeError('Server digest mismatch: '+e['asset'])
      return e['asset'],'uploaded_verified'
     remote=assets(h,release['id']).get(e['asset'])
     if matches(remote,e):return e['asset'],'reconciled_verified'
     if remote:raise RuntimeError('Incomplete remote asset; preserve for review: '+e['asset'])
     r.raise_for_status()
    except requests.RequestException:
     remote=assets(h,release['id']).get(e['asset'])
     if matches(remote,e):return e['asset'],'reconciled_verified'
     if attempt==3:raise
     time.sleep(2*(attempt+1))
  with concurrent.futures.ThreadPoolExecutor(max_workers=a.workers) as pool:
   for i,result in enumerate(pool.map(upload,new),1):print('ASSET',i,len(new),*result,flush=True)
  mp=REPORT/'weights_manifest.json';e=dict(asset='weights_manifest.json',bytes=mp.stat().st_size,sha256=digest(mp));current=assets(h,release['id'])
  if not matches(current.get(e['asset']),e):
   if e['asset'] in current:
    old=current[e['asset']]
    if not a.previous_manifest_sha or old.get('digest')!='sha256:'+a.previous_manifest_sha or m['tag']!='full636-815cc37':raise RuntimeError('Manifest differs; an explicit verified previous digest is required')
    r=requests.delete(f'https://api.github.com/repos/{REPO}/releases/assets/{old["id"]}',headers=h,timeout=(30,120));r.raise_for_status()
   with mp.open('rb') as f:
    r=requests.post(release['upload_url'].split('{')[0]+'?name=weights_manifest.json',headers={**h,'Content-Type':'application/json'},data=f,timeout=(30,120));r.raise_for_status();assert matches(r.json(),e)
 remote=assets(h,release['id']);assert set(remote)=={e['asset'] for e in new}|{'weights_manifest.json'}
 for e in new:assert matches(remote.get(e['asset']),e),e['asset']
 for tag in {e['tag'] for e in m['files'] if e['tag']!=m['tag']}:
  rr=next(r for r in allreleases if r['tag_name']==tag);aa=assets(h,rr['id'])
  for e in m['files']:
   if e['tag']==tag:assert matches(aa.get(e['asset']),e)
 assert remote['weights_manifest.json']['digest']=='sha256:'+digest(REPORT/'weights_manifest.json')
 if a.action=='publish':release=api(h,f'/releases/{release["id"]}','PATCH',dict(draft=False))
 result=dict(status='PASS',release=release['html_url'],draft=release['draft'],unique_weights=len(m['files']),checkpoint_paths=m['checkpoint_paths'],total_unique_bytes=m['total_unique_bytes'],server_size_and_sha256_match=True,assets=len(remote))
 (REPORT/'weights_upload_verification.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf8');print('VERIFIED',json.dumps(result),flush=True)
if __name__=='__main__':main()
