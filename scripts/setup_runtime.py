"""Prepare an isolated local CPU runtime and official model assets."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import urllib.request
import venv


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()


def download(url, path, expected=None):
    if path.is_file() and expected and digest(path) == expected:
        print(f'Reusing verified {path.name}', flush=True)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + '.partial')
    print(f'Downloading {path.name}', flush=True)
    request = urllib.request.Request(url, headers={'User-Agent': 'speaker-isolation-skill/1.0'})
    with urllib.request.urlopen(request, timeout=90) as response, partial.open('wb') as output:
        while chunk := response.read(1024 * 1024):
            output.write(chunk)
    actual = digest(partial)
    if expected and actual != expected:
        raise ValueError(f'SHA-256 mismatch for {path.name}; keep existing files and inspect upstream changes')
    os.replace(partial, path)
    print(f'Validated download: {path.name} ({path.stat().st_size} bytes)', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--skip-install', action='store_true')
    args = parser.parse_args()
    root = args.root.resolve()
    skill = Path(__file__).resolve().parents[1]
    if root == skill or skill in root.parents:
        raise ValueError('Keep runtime/model assets outside the shareable skill folder')
    machine = platform.machine().lower()
    system = platform.system()
    platforms = {
        ('Windows', 'amd64'): 'x86_64-pc-windows-msvc.exe',
        ('Windows', 'x86_64'): 'x86_64-pc-windows-msvc.exe',
        ('Linux', 'x86_64'): 'x86_64-unknown-linux-musl',
        ('Linux', 'aarch64'): 'aarch64-unknown-linux-gnu',
        ('Darwin', 'arm64'): 'aarch64-apple-darwin',
        ('Darwin', 'x86_64'): 'x86_64-apple-darwin',
    }
    target = platforms.get((system, machine))
    if not target:
        raise RuntimeError(f'No configured official DeepFilterNet binary for {system}/{machine}')
    root.mkdir(parents=True, exist_ok=True)
    if not args.skip_install:
        if sys.version_info[:2] != (3, 12):
            raise RuntimeError('Use Python 3.12 for this tested runtime setup')
        env = root / 'venv'
        python = env / ('Scripts/python.exe' if system == 'Windows' else 'bin/python')
        if not python.exists():
            venv.EnvBuilder(with_pip=True).create(env)
        torch_command = [str(python), '-m', 'pip', 'install', 'torch==2.6.0']
        if system in ('Windows', 'Linux'):
            torch_command += ['--index-url', 'https://download.pytorch.org/whl/cpu']
        subprocess.run(torch_command, check=True)
        subprocess.run([str(python), '-m', 'pip', 'install', 'numpy>=1.26,<3',
                        'av>=14,<20', 'soundfile>=0.12,<0.15', 'einops==0.8.2',
                        'rotary-embedding-torch==0.8.3', 'torchinfo==1.8.0'], check=True)
    manifest = json.loads((skill / 'references/downloads.json').read_text(encoding='utf-8'))
    downloaded = []
    for asset in manifest['assets']:
        path = root / asset['path']
        download(asset['url'], path, asset['sha256'])
        downloaded.append({'path':asset['path'], 'sha256':digest(path), 'url':asset['url']})
    filename = f'deep-filter-0.5.6-{target}'
    url = 'https://github.com/Rikorose/DeepFilterNet/releases/download/v0.5.6/' + filename
    binary = root / 'DeepFilterNet' / ('deep-filter.exe' if system == 'Windows' else 'deep-filter')
    download(url, binary, manifest['deepfilter_sha256'].get(target))
    if system != 'Windows':
        binary.chmod(binary.stat().st_mode | 0o111)
    downloaded.append({'path':str(binary.relative_to(root)), 'sha256':digest(binary), 'url':url})
    (root / 'download-record.json').write_text(json.dumps(downloaded, indent=2), encoding='utf-8')
    print('Ready. Pass this root as --assets when running isolate.py.', flush=True)


if __name__ == '__main__':
    main()
