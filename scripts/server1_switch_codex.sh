#!/usr/bin/env bash
# Change the manager only; research jobs and the original checkout's experiment code are untouched.
set -euo pipefail
case "$(hostname)" in *-com-loop-*) ;; *) echo "서버 1(loop) 전용"; exit 1;; esac
cd "$HOME/R2E"
mkdir -p agent/logs
exec 9>agent/.codex_switch.lock
flock -n 9 || { echo "Another switch installer is active"; exit 1; }
if command -v codex >/dev/null; then
  codex_bin=$(command -v codex)
else
  codex_bin="$HOME/.local/share/r2e-codex/node_modules/.bin/codex"
  if [ ! -x "$codex_bin" ]; then
    if command -v npm >/dev/null; then
      npm install --prefix "$HOME/.local/share/r2e-codex" @openai/codex
    else
      echo "npm 없이 공식 Linux Codex 실행 파일을 설치합니다."
      .venv/bin/python - "$codex_bin" <<'INSTALL_PY'
import os, platform, shutil, subprocess, sys, tarfile, tempfile, urllib.request
from pathlib import Path
if sys.platform!='linux':raise SystemExit('Native fallback is Linux only; nothing switched')
arch={'x86_64':'x86_64','amd64':'x86_64','aarch64':'aarch64','arm64':'aarch64'}.get(platform.machine().lower())
if arch is None:raise SystemExit('Unsupported architecture; nothing switched')
name=f'codex-{arch}-unknown-linux-musl'
url=f'https://github.com/openai/codex/releases/latest/download/{name}.tar.gz'
dest=Path(sys.argv[1]);dest.parent.mkdir(parents=True,exist_ok=True)
with tempfile.TemporaryDirectory(prefix='r2e-codex-install-') as tmp:
    archive=Path(tmp)/'codex.tar.gz'
    if shutil.which('curl'):
        subprocess.run(['curl','-fL','--retry','3','--connect-timeout','20','--max-time','240','-o',str(archive),url],check=True)
    else:
        request=urllib.request.Request(url,headers={'User-Agent':'R2E-Codex-Installer'})
        with urllib.request.urlopen(request,timeout=240) as response,archive.open('wb') as f:shutil.copyfileobj(response,f)
    with tarfile.open(archive,'r:gz') as tar:
        matches=[m for m in tar.getmembers() if m.isfile() and Path(m.name).name==name]
        if len(matches)!=1:raise SystemExit('Official archive binary not found; nothing switched')
        source=tar.extractfile(matches[0]);assert source is not None
        staged=dest.with_name(dest.name+'.installing')
        with staged.open('wb') as f:shutil.copyfileobj(source,f)
        staged.chmod(0o755);staged.replace(dest)
print('Installed official Codex binary:',dest)
INSTALL_PY
    fi
  fi
fi
"$codex_bin" --version
if ! "$codex_bin" login status; then
  echo "서버 Codex 로그인: 표시되는 주소를 브라우저에서 열어 인증하세요."
  "$codex_bin" login --device-auth
fi
"$codex_bin" login status
# Fetch into a dedicated ref; never rebase, reset, stash, or checkout the live experiment folder.
git fetch https://github.com/jeehoo0507/R2E.git main:refs/remotes/codex-bootstrap/main
bootstrap_ref=$(git rev-parse refs/remotes/codex-bootstrap/main)
git show "$bootstrap_ref:scripts/codex_install_helpers.py" > scripts/codex_install_helpers.py.installing
mv scripts/codex_install_helpers.py.installing scripts/codex_install_helpers.py
.venv/bin/python scripts/codex_install_helpers.py --codex "$codex_bin"
bootstrap_backup="agent/backups/codex_$(date -u +%Y%m%d_%H%M%S)_$$"
mkdir -p "$bootstrap_backup"
export R2E_CODEX_BOOTSTRAP_REF="$bootstrap_ref"
export R2E_CODEX_BACKUP="$bootstrap_backup"
.venv/bin/python - <<'PY'
import json, os, shutil, signal, subprocess, time
from pathlib import Path
root=Path.cwd(); work=Path.home()/'R2E-codex';backup=root/os.environ['R2E_CODEX_BACKUP'];ref=os.environ['R2E_CODEX_BOOTSTRAP_REF']
branch='server1-pkg-base-20261106'
if not work.exists():
    subprocess.run(['git','worktree','add','-b',branch,str(work),ref],check=True)
else:
    top=subprocess.check_output(['git','rev-parse','--show-toplevel'],cwd=work,text=True).strip()
    current=subprocess.check_output(['git','branch','--show-current'],cwd=work,text=True).strip()
    if Path(top)!=work or current!=branch:raise SystemExit('Unexpected worktree; nothing stopped')
# Copy immutable inputs; do not symlink live runs/checkpoints.
inputs=list((root/'runs/pool_v1').glob('*.pkl'))+list((root/'data/teacher').glob('*.npz'))
inputs+=list((root/'reports/peak_mask_priority_20261106').glob('*.json'))
inputs+=list((root/'reports/peak_mask_diag_20261106').glob('*.json'))
for name in ('r2pk','r2pk_s1','r2pk_s2'):
    inputs+=list((root/f'runs/2026-10-29/{name}_rl').glob('best.pt'))
for source in inputs:
    target=work/source.relative_to(root)
    if not target.exists():target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
venv=work/'.venv'
if not venv.exists():venv.symlink_to(root/'.venv',target_is_directory=True)
# Verify the old loop is idle before replacing its launcher. Never signal any descendant.
def managers():
    result=[]
    for p in Path('/proc').glob('[0-9]*/cmdline'):
        try:
            args=[v.decode(errors='replace') for v in p.read_bytes().split(b'\0') if v]
            if len(args)>=2 and Path(args[0]).name=='bash' and args[1]=='agent/loop.sh' and (p.parent/'cwd').resolve()==root:
                result.append(int(p.parent.name))
        except OSError:pass
    return result

def idle(pid):
    try:children=(Path(f'/proc/{pid}/task/{pid}/children').read_text()).split()
    except FileNotFoundError:return True
    for child in children:
        try:args=Path(f'/proc/{child}/cmdline').read_bytes().split(b'\0')
        except FileNotFoundError:continue
        if args and args[0] and Path(os.fsdecode(args[0])).name!='sleep':return False
    return True
active_codex=False
for p in Path('/proc').glob('[0-9]*/cmdline'):
    try:
        args=[os.fsdecode(x) for x in p.read_bytes().split(b'\0') if x]
        if len(args)>1 and args[1]=='agent/codex_loop.sh' and (p.parent/'cwd').resolve()==root:active_codex=True
    except OSError:pass
if active_codex:
    raise SystemExit('Codex manager already selected and running; no process stopped')
for _ in range(12):
    old=managers()
    if all(idle(pid) for pid in old):break
    print('Waiting for the active management round to finish; research jobs unchanged',flush=True);time.sleep(5)
else:raise SystemExit('Management round still active; retry installer later. No process stopped.')
files=('agent/loop.sh','agent/codex_loop.sh','agent/CODEX_PKG_TASK.md','scripts/codex_pkg_round.py')
for name in files:
    dest=root/name
    if dest.exists():
        saved=backup/name;saved.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(dest,saved)
    data=subprocess.check_output(['git','show',f'{ref}:{name}'])
    temp=dest.with_suffix(dest.suffix+'.installing');temp.write_bytes(data);temp.replace(dest)
    # The isolated worktree gets its own new prompt, without changing experiment modules.
    if name=='agent/CODEX_PKG_TASK.md':shutil.copy2(dest,work/name)
provider=root/'agent/PROVIDER'
if provider.exists():shutil.copy2(provider,backup/'PROVIDER')
provider.write_text('codex\n')
for pid in old:
    # Freeze only the shell, recheck that it owns no active command, terminate shell, then continue it.
    try:
        os.kill(pid,signal.SIGSTOP)
        time.sleep(.1)
        if not idle(pid):
            os.kill(pid,signal.SIGCONT)
            raise SystemExit('Round became active; old manager resumed. Retry later.')
        os.kill(pid,signal.SIGTERM);os.kill(pid,signal.SIGCONT)
    except ProcessLookupError:pass
    for _ in range(50):
        if not Path(f'/proc/{pid}').exists():break
        time.sleep(.1)
receipt=dict(provider='codex',worktree=str(work),branch=branch,backup=str(backup),installed_ref=ref,
             copied_input_files=len(inputs),terminated_idle_manager_shells=old)
(root/'agent/CODEX_INSTALL.json').write_text(json.dumps(receipt,indent=2))
print(json.dumps(receipt,indent=2),flush=True)
PY
if [ -f agent/STOP ]; then
  echo "agent/STOP이 있어 관리자 시작은 보류합니다. 기존 실험에는 영향 없습니다."
  exit 0
fi
.venv/bin/python scripts/bg.py agent/logs/codex_launcher.log 'bash agent/loop.sh'
echo "Codex manager launch requested. Check agent/logs/codex_loop.log and agent/CODEX_STATUS.json."
echo "Existing training/evaluation processes were not signalled or restarted."
