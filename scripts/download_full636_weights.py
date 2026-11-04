"""Plan or restore content-deduplicated full636 Release weights with SHA-256 checks."""
import argparse,hashlib,json,os,urllib.request,tempfile,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
 return h.hexdigest()
def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--arm',choices=['eqgpt_reference','ast_bpe_280','ast_bpe_315','ast_bpe_350']);ap.add_argument('--case');ap.add_argument('--download',action='store_true');a=ap.parse_args()
 m=json.loads((ROOT/'reports/full636_run/weights_manifest.json').read_text());selected=[]
 for e in m['files']:
  paths=[]
  for rel in e['paths']:
   p=Path(rel)
   if p.is_absolute() or '..' in p.parts or p.parts[:3]!=('experiments','full','636'):raise ValueError('Unsafe weight path')
   if a.arm and a.arm not in p.parts:continue
   if a.case and a.case not in p.parts and 'shared_pretrain' not in p.parts:continue
   paths.append(ROOT/p)
  if paths:selected.append((e,paths))
 if not selected:raise ValueError('No matching checkpoint paths')
 print(json.dumps(dict(unique_weights=len(selected),checkpoint_paths=sum(len(p) for _,p in selected),bytes=sum(e['bytes'] for e,_ in selected),download=a.download),indent=2))
 if not a.download:return
 for i,(e,paths) in enumerate(selected,1):
  existing=[p for p in paths if p.exists()]
  for p in existing:
   if p.stat().st_size!=e['bytes'] or sha(p)!=e['sha256']:raise ValueError('Existing weight differs; refusing replacement: '+str(p))
  target=existing[0] if existing else paths[0];target.parent.mkdir(parents=True,exist_ok=True)
  if not existing:
   url=f'https://github.com/{m["repository"]}/releases/download/{e["tag"]}/{e["asset"]}?download=1'
   with tempfile.NamedTemporaryFile(dir=target.parent,suffix='.part',delete=False) as f:
    temporary=Path(f.name)
    request=urllib.request.Request(url,headers={'User-Agent':'tokenizer-pde-discovery-weight-restore'})
    with urllib.request.urlopen(request,timeout=120) as response:shutil.copyfileobj(response,f,8*1024*1024)
   if temporary.stat().st_size!=e['bytes'] or sha(temporary)!=e['sha256']:raise ValueError('Downloaded weight checksum mismatch: '+e['asset'])
   os.replace(temporary,target)
  for p in paths:
   if not p.exists():
    p.parent.mkdir(parents=True,exist_ok=True)
    try:os.link(target,p)
    except OSError:shutil.copyfile(target,p)
  print(i,len(selected),'verified',e['asset'],flush=True)
if __name__=='__main__':main()
