"""Install official Linux runtime companions matching the installed native Codex CLI."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import tarfile
import tempfile
import urllib.request


def unpack(archive,destination,expected):
    """Copy one regular executable only, never extract archive paths or links."""
    with tarfile.open(archive,'r:gz') as tar:
        matches=[m for m in tar.getmembers() if m.isfile() and Path(m.name).name in expected]
        if len(matches)!=1:raise ValueError(f'Expected one runtime executable, found {[m.name for m in matches]}')
        source=tar.extractfile(matches[0]);assert source is not None
        destination.parent.mkdir(parents=True,exist_ok=True)
        staged=destination.with_name(destination.name+'.installing')
        with staged.open('wb') as f:shutil.copyfileobj(source,f)
        staged.chmod(0o755);staged.replace(destination)


def main():
    p=argparse.ArgumentParser();p.add_argument('--codex',required=True);a=p.parse_args()
    binary=Path(a.codex).resolve()
    version=subprocess.check_output([str(binary),'--version'],text=True).strip()
    match=re.search(r'codex-cli\s+(\S+)',version)
    if not match:raise SystemExit(f'Cannot identify CLI version: {version}')
    tag='rust-v'+match.group(1)
    arch={'x86_64':'x86_64','amd64':'x86_64','aarch64':'aarch64','arm64':'aarch64'}.get(platform.machine().lower())
    if platform.system()!='Linux' or arch is None:raise SystemExit('Helper installer supports Linux x86_64/aarch64')
    request=urllib.request.Request(f'https://api.github.com/repos/openai/codex/releases/tags/{tag}',headers={'User-Agent':'R2E-Codex-Installer'})
    with urllib.request.urlopen(request,timeout=30) as response:release=json.load(response)
    assets={x['name']:x for x in release['assets']}
    for executable in ('codex-code-mode-host','bwrap'):
        name=f'{executable}-{arch}-unknown-linux-musl'
        asset=assets[name+'.tar.gz']
        destination=binary.parent/executable
        with tempfile.TemporaryDirectory(prefix='r2e-codex-helper-') as tmp:
            archive=Path(tmp)/'runtime.tar.gz'
            if shutil.which('curl'):
                subprocess.run(['curl','-fL','--retry','3','--connect-timeout','20','--max-time','240','-o',str(archive),asset['browser_download_url']],check=True)
            else:
                req=urllib.request.Request(asset['browser_download_url'],headers={'User-Agent':'R2E-Codex-Installer'})
                with urllib.request.urlopen(req,timeout=240) as response,archive.open('wb') as f:shutil.copyfileobj(response,f)
            digest=asset.get('digest')
            if digest and digest.startswith('sha256:'):
                if hashlib.sha256(archive.read_bytes()).hexdigest()!=digest[7:]:raise ValueError('Official asset digest mismatch')
            unpack(archive,destination,{name,executable})
        print(f'Installed {executable} matching {tag}: {destination}',flush=True)
    print('Runtime companions installed. No experiment process was changed.',flush=True)


if __name__=='__main__':main()
