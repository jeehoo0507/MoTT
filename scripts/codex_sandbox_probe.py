"""Test existing Codex isolation, optionally inside an unprivileged user/mount namespace.

Keeps workspace-write, bwrap, and seccomp; never uses sudo, changes host policy, or stops jobs.
Only marks a backend ready after an in-workspace write succeeds and an outside write is denied.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import uuid

ROOT=Path(__file__).resolve().parents[1]
WORK=Path.home()/'R2E-codex'
CODE='''import errno,pathlib,sys
inside,outside,nonce=pathlib.Path(sys.argv[1]),pathlib.Path(sys.argv[2]),sys.argv[3]
inside.write_text(nonce)
try:
    outside.write_text(nonce)
except OSError as error:
    if error.errno not in (errno.EACCES,errno.EPERM,errno.EROFS):raise
else:
    raise SystemExit("Outside workspace write unexpectedly permitted")
print("R2E_SANDBOX_OK:"+nonce)
'''


def namespace_prefix():
    command=shutil.which('unshare')
    return [command,'--user','--map-root-user','--mount','--propagation','private','--'] if command else None


def probe(codex,prefix,work,outside_root,python=sys.executable):
    nonce=uuid.uuid4().hex
    inside=Path(work)/f'.codex_inside_probe_{nonce}'
    outside=Path(outside_root)/f'.codex_outside_probe_{nonce}'
    cmd=prefix+[str(codex),'--sandbox','workspace-write','--cd',str(work),'sandbox','--',
                python,'-c',CODE,str(inside),str(outside),nonce]
    try:
        env=dict(os.environ,PATH=str(Path(codex).resolve().parent)+os.pathsep+os.environ.get('PATH',''))
        result=subprocess.run(cmd,cwd=work,capture_output=True,text=True,timeout=30,env=env)
        allowed=inside.exists() and inside.read_text()==nonce
        denied=not outside.exists()
        success=result.returncode==0 and allowed and denied and ('R2E_SANDBOX_OK:'+nonce) in result.stdout
        return dict(success=success,rc=result.returncode,inside_write_verified=allowed,
                    outside_write_denied=denied,stdout=result.stdout[-5000:],stderr=result.stderr[-5000:],prefix=prefix)
    except (OSError,subprocess.TimeoutExpired) as error:
        return dict(success=False,error=str(error),prefix=prefix)
    finally:
        inside.unlink(missing_ok=True);outside.unlink(missing_ok=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--codex',required=True);a=p.parse_args()
    if not WORK.is_dir():raise SystemExit('Missing isolated worktree; nothing changed')
    attempts=[]
    for prefix in ([],namespace_prefix()):
        if prefix is None:continue
        item=probe(a.codex,prefix,WORK,ROOT/'agent');attempts.append(item)
        print(json.dumps(item,ensure_ascii=False,indent=2),flush=True)
        if item['success']:
            payload=dict(verified=True,prefix=prefix,policy='workspace-write',
                         checked_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
                         inside_write_verified=True,outside_write_denied=True)
            target=ROOT/'agent/CODEX_SANDBOX_BACKEND.json'
            tmp=target.with_suffix('.tmp');tmp.write_text(json.dumps(payload,indent=2)+'\n');tmp.replace(target)
            (ROOT/'agent/CODEX_RETRY_SANDBOX').touch()
            (ROOT/'agent/CODEX_SANDBOX_DIAG.json').write_text(json.dumps(attempts,indent=2)+'\n')
            print('SANDBOX VERIFIED: normal workspace policy retained; ready to start research round',flush=True)
            return
    (ROOT/'agent/CODEX_SANDBOX_DIAG.json').write_text(json.dumps(attempts,indent=2)+'\n')
    (ROOT/'agent/CODEX_STATUS.json').write_text(json.dumps(dict(stage='blocked_sandbox',model_invoked=False,
        reason='Neither native nor isolated-user-namespace launch can enforce the workspace policy',
        diagnostics=str(ROOT/'agent/CODEX_SANDBOX_DIAG.json')),indent=2))
    raise SystemExit('Host denies required namespace/mount operations. No policy disabled; no research job changed.')


if __name__=='__main__':main()
