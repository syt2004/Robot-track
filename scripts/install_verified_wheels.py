"""Install checksum-verified locked wheels into one project-owned environment.

--ready-only explicitly permits a partial environment; it does not certify
imports, CUDA, dependency consistency or model readiness. No network is used.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
from check_wheelhouse import audit


def install(root,component,output,ready_only=False):
    root=Path(root).resolve(strict=True)
    output=Path(output).absolute()
    if output.exists():
        raise FileExistsError(output)
    names={'trackon':('trackon312','pylock.trackon-full.toml'),
           'track4world':('track4world311','pylock.track4world-built.toml')}
    environment,lock=names[component]
    report=audit(root/'environments'/lock,root/'.wheelhouse')
    report.update(component=component,mode='ready_only' if ready_only else 'all_locked_wheels',
                  environment_ready=False,state='checking',imports_verified=False,cuda_verified=False)
    pending=[p for p in report['packages'] if p['state']!='verified']
    if any(p['state']=='mismatch' for p in pending):
        raise ValueError('Mismatched wheel; refusing installation')
    if pending and not ready_only:
        raise ValueError('Not all locked wheels verified; use --ready-only only for explicit partial installation')
    wheels=[str(root/'.wheelhouse'/p['file']) for p in report['packages'] if p['state']=='verified']
    if not wheels:
        raise ValueError('No verified wheels to install')
    python=root/'.envs'/environment/'bin/python'  # Never dereference the venv symlink.
    uv=root/'.envs/trackon/bin/uv'
    if not python.is_file() or not uv.is_file():
        raise FileNotFoundError('Project interpreter/uv missing')
    output.mkdir(parents=True,exist_ok=False)
    def save():
        temporary=output/'install.pending.json'
        temporary.write_text(json.dumps(report,indent=2)+'\n')
        temporary.replace(output/'install.json')
    command=[str(uv),'pip','install','--python',str(python),'--no-index','--no-deps',*wheels]
    report.update(state='installing',command=command,pending_packages=[p['name'] for p in pending])
    save()
    env=os.environ.copy()
    env['UV_CACHE_DIR']=str(root/'.cache/uv')
    try:
        with (output/'install.log').open('xb') as log:
            result=subprocess.run(command,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT)
        report.update(returncode=result.returncode,state='installed_partial' if pending else 'installed_unchecked')
        if result.returncode:
            raise RuntimeError('uv installation failed; inspect '+str(output/'install.log'))
        save()
    except BaseException as error:
        report.update(state='failed',error=type(error).__name__+': '+str(error))
        save()
        raise
    return output/'install.json'


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',default='.')
    parser.add_argument('--component',choices=('trackon','track4world'),required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--ready-only',action='store_true')
    args=parser.parse_args()
    print(install(args.root,args.component,args.output,args.ready_only))
