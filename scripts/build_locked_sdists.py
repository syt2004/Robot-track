"""Build missing wheels only from the exact source archives in a pylock.

Original locks are never rewritten. Each build has a separate directory/log and
records source/produced hashes; generated wheels still need runtime import checks.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import tomllib
from urllib.parse import urlsplit,unquote
from download_verified import download,file_digest


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('lock')
    parser.add_argument('--python',required=True)
    parser.add_argument('--uv',default='.envs/trackon/bin/uv')
    parser.add_argument('--output',required=True,help='New build workspace, retained for provenance')
    parser.add_argument('--wheelhouse',default='.wheelhouse')
    args=parser.parse_args()
    lock_path=Path(args.lock).resolve(strict=True)
    lock=tomllib.loads(lock_path.read_text())
    packages=[p for p in lock['packages'] if not p.get('wheels')]
    output=Path(args.output).resolve()
    output.mkdir(parents=True,exist_ok=False)
    wheelhouse=Path(args.wheelhouse).resolve()
    wheelhouse.mkdir(parents=True,exist_ok=True)
    uv=str(Path(args.uv).resolve(strict=True))
    python=os.path.abspath(args.python)
    if not Path(python).is_file():
        raise FileNotFoundError(python)
    environment=os.environ.copy()
    environment.update(UV_HTTP_CONNECT_TIMEOUT='120',UV_HTTP_TIMEOUT='120')
    report=dict(state='running',source_lock_sha256=file_digest(lock_path),builds=[])
    def save():
        pending=output/'manifest.pending.json'
        pending.write_text(json.dumps(report,indent=2))
        pending.replace(output/'manifest.json')
    save()
    try:
        shutil.copyfile(lock_path,output/'source-lock.toml')
        for package in packages:
            source=package['sdist']
            filename=unquote(urlsplit(source['url']).path.rsplit('/',1)[-1])
            if Path(filename).name!=filename or not filename.endswith(('.tar.gz','.zip')):
                raise ValueError('Unsafe source filename')
            archive=download(source['url'],output/filename,source['hashes']['sha256'])
            build_dir=output/(package['name']+'-'+package['version'])
            build_dir.mkdir()
            log=build_dir/'build.log'
            command=[uv,'build',str(archive),'--wheel','--python',python,'--out-dir',str(build_dir),
                     '--no-create-gitignore','--default-index','https://pypi.org/simple']
            with log.open('xb') as stream:
                subprocess.run(command,env=environment,stdout=stream,stderr=subprocess.STDOUT,check=True)
            wheels=list(build_dir.glob('*.whl'))
            if len(wheels)!=1:
                raise ValueError('Expected exactly one source-built wheel')
            wheel=wheels[0]
            digest=file_digest(wheel)
            destination=wheelhouse/wheel.name
            if destination.exists():
                if file_digest(destination)!=digest:
                    raise ValueError('Existing wheel differs; refusing overwrite')
            else:
                with wheel.open('rb') as src,destination.open('xb') as dst:
                    shutil.copyfileobj(src,dst)
            report['builds'].append(dict(name=package['name'],version=package['version'],
                                        source_url=source['url'],source_sha256=source['hashes']['sha256'],
                                        wheel=wheel.name,wheel_sha256=digest,command=command))
            save()
            print('BUILT '+wheel.name,flush=True)
        report['state']='complete'
        save()
    except BaseException as error:
        report.update(state='failed',error=type(error).__name__+': '+str(error))
        save()
        raise


if __name__=='__main__':
    main()
