"""Reconstruct the archived full636 runtime in a new, isolated directory."""
import argparse,hashlib,json,subprocess,tarfile,tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--output',type=Path,required=True);args=ap.parse_args();out=args.output.resolve()
 if out.exists() and any(out.iterdir()):raise FileExistsError('Use a new or empty output directory')
 report=ROOT/'reports/full636_run';m=json.loads((report/'runtime_source_manifest.json').read_text());out.mkdir(parents=True,exist_ok=True)
 if subprocess.run(['git','cat-file','-e',m['base_commit']+'^{commit}'],cwd=ROOT,capture_output=True).returncode:
  subprocess.run(['git','fetch','origin',m['base_commit']],cwd=ROOT,check=True)
 with tempfile.TemporaryFile() as f:
  subprocess.run(['git','-c','core.autocrlf=false','archive',m['base_commit']],cwd=ROOT,stdout=f,check=True);f.seek(0)
  with tarfile.open(fileobj=f) as t:t.extractall(out,filter='data')
 # The archived CUDA run normalized the checkout's UTF-8 text to LF.
 # Apply this only to the new isolated copy, never the user's checkout.
 for p in out.rglob('*'):
  if not p.is_file() or p.is_symlink():continue
  if 'eqgpt_original' in p.relative_to(out).parts:continue
  raw=p.read_bytes()
  if b'\x00' in raw or b'\r\n' not in raw:continue
  try:raw.decode('utf8')
  except UnicodeDecodeError:continue
  p.write_bytes(raw.replace(b'\r\n',b'\n'))
 subprocess.run(['git','-c','core.autocrlf=false','apply','--unsafe-paths','--directory='+str(out),str(report/'runtime.patch')],cwd=ROOT,check=True)
 for e in m['source_files']:
  p=out/e['path'];assert hashlib.sha256(p.read_bytes()).hexdigest()==e['sha256'],e['path']
 print(json.dumps(dict(status='PASS',output=str(out),verified_source_files=len(m['source_files']))))
if __name__=='__main__':main()
