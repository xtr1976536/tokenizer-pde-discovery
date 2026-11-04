"""Upload all full212 checkpoints to a draft Release and verify server digests.

Run from the repository root. Re-running skips assets only when size and digest
match. Never modifies checkpoints or starts an experiment.
"""
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time

REPO = 'xtr1976536/tokenizer-pde-discovery'
TAG = 'full212-815cc37'
ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / 'reports/full212_run'


def gh(*args):
    return subprocess.check_output(['gh', *args], text=True)


def assets():
    releases = json.loads(gh('api', '--paginate', '--slurp', f'repos/{REPO}/releases?per_page=100'))
    release = next((r for page in releases for r in page if r['tag_name'] == TAG), None)
    if release is None:
        raise RuntimeError(f'Release {TAG} is missing')
    rows = json.loads(gh('api', '--paginate', '--slurp', f'repos/{REPO}/releases/{release["id"]}/assets?per_page=100'))
    return {a['name']: a for page in rows for a in page}


def main():
    os.chdir(ROOT)
    groups = {}
    for p in sorted((ROOT / 'experiments/full/212').rglob('*.pt')):
        st = p.stat()
        groups.setdefault((st.st_dev, st.st_ino), []).append(p)
    entries = []
    for paths in groups.values():
        p = paths[0]
        h = hashlib.sha256()
        with p.open('rb') as f:
            for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
                h.update(chunk)
        name = '__'.join(p.relative_to(ROOT / 'experiments/full/212').parts)
        entries.append(dict(asset=name, bytes=p.stat().st_size, sha256=h.hexdigest(),
                            paths=[x.relative_to(ROOT).as_posix() for x in paths]))
    manifest = dict(repository=REPO, tag=TAG, training_commit='815cc37adcdd714c9e9bd547a0561bb75f82901c',
                    scale=212, unique_assets=len(entries), checkpoint_paths=sum(len(x['paths']) for x in entries),
                    total_unique_bytes=sum(x['bytes'] for x in entries), files=entries)
    mp = REPORT / 'weights_manifest.json'
    mp.write_text(json.dumps(manifest, indent=2) + '\n')
    print('MANIFEST', len(entries), manifest['total_unique_bytes'], flush=True)
    existing = assets()
    def matches(remote, entry):
        return remote and remote['size'] == entry['bytes'] and remote.get('digest') == 'sha256:' + entry['sha256']
    with tempfile.TemporaryDirectory(prefix='full212-release-') as tmp:
        def upload(entry):
            name = entry['asset']
            if matches(existing.get(name), entry):
                return name, 'already_verified'
            link = Path(tmp) / (name.replace('/', '--'))
            link.symlink_to(ROOT / entry['paths'][0])
            for attempt in range(4):
                result = subprocess.run(['gh', 'release', 'upload', TAG, f'{link}#{name}', '--repo', REPO, '--clobber'],
                                        capture_output=True, text=True)
                if result.returncode == 0:
                    return name, 'uploaded'
                if attempt == 3:
                    raise RuntimeError(name + ': ' + result.stderr[-1500:])
                time.sleep(3 * (attempt + 1))
        completed = 0
        with concurrent.futures.ThreadPoolExecutor(max_workers=12) as pool:
            futures = [pool.submit(upload, e) for e in entries]
            for future in concurrent.futures.as_completed(futures):
                name, state = future.result()
                completed += 1
                print('ASSET', completed, '/', len(entries), state, name, flush=True)
    remote = assets()
    for e in entries:
        assert matches(remote.get(e['asset']), e), e['asset']
    gh('release', 'upload', TAG, str(mp), '--repo', REPO, '--clobber')
    verification = dict(status='PASS', assets=len(entries), checkpoint_paths=manifest['checkpoint_paths'],
                        total_bytes=manifest['total_unique_bytes'],
                        verification='GitHub server-reported SHA-256 digest and size match each local checkpoint',
                        release=f'https://github.com/{REPO}/releases/tag/{TAG}')
    (REPORT / 'weights_upload_verification.json').write_text(json.dumps(verification, indent=2) + '\n')
    gh('release', 'edit', TAG, '--repo', REPO, '--draft=false')
    print('COMPLETE_VERIFIED', json.dumps(verification), flush=True)


if __name__ == '__main__':
    main()
