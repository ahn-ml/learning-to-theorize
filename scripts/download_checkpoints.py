#!/usr/bin/env python3
"""Download the pretrained observation models (and the Arithmetic pretraining
episodes) and verify their recorded checksums."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tarfile
import tempfile
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
TASKS = ('gridworld', 'arithmetic_factorization', 'image_editing')


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def extract_verified(archive: Path, output: Path, files: list[dict]) -> None:
    """Reject unlisted files/links and validate contents before installing them."""
    expected = {row['path']: row for row in files}
    with tempfile.TemporaryDirectory(dir=output) as temporary:
        staging = Path(temporary)
        with tarfile.open(archive) as bundle:
            seen = set()
            for member in bundle.getmembers():
                path = Path(member.name)
                if path.is_absolute() or '..' in path.parts or not (member.isdir() or member.isfile()):
                    raise ValueError(f'unsafe archive member: {member.name}')
                if member.isdir():
                    continue
                if member.name not in expected or member.name in seen:
                    raise ValueError(f'unexpected archive member: {member.name}')
                seen.add(member.name)
                target = staging / path
                target.parent.mkdir(parents=True, exist_ok=True)
                source = bundle.extractfile(member)
                with source, target.open('wb') as destination:
                    shutil.copyfileobj(source, destination)
                row = expected[member.name]
                if target.stat().st_size != row['bytes'] or sha256(target) != row['sha256']:
                    raise ValueError(f'file checksum mismatch: {member.name}')
        if seen != set(expected):
            raise ValueError('archive is missing required files')
        # Check every destination before installing any files from this archive.
        for name, row in expected.items():
            target = output / name
            if output.resolve() not in target.resolve().parents:
                raise ValueError(f'destination escapes output directory: {name}')
            if target.exists() and (not target.is_file() or sha256(target) != row['sha256']):
                raise ValueError(f'existing file differs: {target}')
        for name in expected:
            target = output / name
            if not target.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(staging / name), target)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', choices=TASKS, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads((ROOT / 'configs/checkpoints.json').read_text())
    base = 'https://github.com/ahn-ml/learning-to-theorize/releases/download/' + manifest['release_tag']
    output = args.output_root.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    for archive in manifest['archives']:
        if not archive['name'].startswith(args.task + '-'):
            continue
        condition = archive['name'][len(args.task) + 1:-4]
        prefix = f'{args.task}/{condition}/'
        files = [row for row in manifest['files'] if row['path'].startswith(prefix)]
        if not files:
            raise ValueError(f'no file manifest for {archive["name"]}')
        if all((output / row['path']).is_file() and sha256(output / row['path']) == row['sha256'] for row in files):
            print(f'Verified existing {args.task}/{condition}')
            continue
        with tempfile.TemporaryDirectory(dir=output) as temporary:
            path = Path(temporary) / archive['name']
            print(f'Downloading {archive["name"]}', flush=True)
            with urlopen(base + '/' + archive['name']) as response, path.open('wb') as destination:
                shutil.copyfileobj(response, destination)
            if path.stat().st_size != archive['bytes'] or sha256(path) != archive['sha256']:
                raise ValueError(f'archive checksum mismatch: {archive["name"]}')
            extract_verified(path, output, files)
    print(f'Verified {args.task} checkpoints in {output}')


if __name__ == '__main__':
    main()
