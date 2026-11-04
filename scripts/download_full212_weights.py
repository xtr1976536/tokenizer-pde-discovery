"""Restore the full212 Release checkpoints with SHA-256 verification.

Requires GitHub CLI. Default is a download plan; --download restores files.
Existing matching files are reused. Existing mismatching files are never replaced.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--arm', choices=['eqgpt_reference', 'ast_bpe_280', 'ast_bpe_315', 'ast_bpe_350'])
    ap.add_argument('--case', help='Exact case directory name; shared pretraining is also restored')
    ap.add_argument('--download', action='store_true')
    args = ap.parse_args()
    manifest = json.loads((ROOT / 'reports/full212_run/weights_manifest.json').read_text())
    selected = []
    for entry in manifest['files']:
        paths = []
        for rel in entry['paths']:
            path = Path(rel)
            if path.is_absolute() or '..' in path.parts or path.parts[:3] != ('experiments', 'full', '212'):
                raise ValueError('Unsafe checkpoint path: ' + rel)
            if args.arm and args.arm not in path.parts:
                continue
            if args.case and args.case not in path.parts and 'shared_pretrain' not in path.parts:
                continue
            paths.append(ROOT / path)
        if paths:
            selected.append((entry, paths))
    if not selected:
        raise ValueError('No matching checkpoint paths')
    print(json.dumps(dict(assets=len(selected), paths=sum(len(p) for _, p in selected),
                          unique_bytes=sum(e['bytes'] for e, _ in selected), download=args.download), indent=2))
    if not args.download:
        return
    for i, (entry, paths) in enumerate(selected, 1):
        existing = [p for p in paths if p.exists()]
        for p in existing:
            if p.stat().st_size != entry['bytes'] or digest(p) != entry['sha256']:
                raise ValueError('Existing checkpoint differs; refusing replacement: ' + str(p))
        target = existing[0] if existing else paths[0]
        target.parent.mkdir(parents=True, exist_ok=True)
        if not existing:
            with tempfile.TemporaryDirectory(prefix='.weights-download-', dir=target.parent) as tmp:
                subprocess.run(['gh', 'release', 'download', manifest['tag'], '--repo', manifest['repository'],
                                '--pattern', entry['asset'], '--dir', tmp], check=True)
                source = Path(tmp) / entry['asset']
                if source.stat().st_size != entry['bytes'] or digest(source) != entry['sha256']:
                    raise ValueError('Downloaded checkpoint failed checksum: ' + entry['asset'])
                os.replace(source, target)
        for p in paths:
            if not p.exists():
                p.parent.mkdir(parents=True, exist_ok=True)
                os.link(target, p)
        print(f'{i}/{len(selected)} verified {entry["asset"]}', flush=True)


if __name__ == '__main__':
    main()
