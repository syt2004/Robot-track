"""Read-only installation gate for the exact wheel selections in a pylock.

Checks the same first wheel as download_wheelhouse.py. Partial files, missing
source-built wheels, and mismatched hashes never count as installation-ready.
"""
import argparse
import json
from pathlib import Path
import tomllib
from urllib.parse import unquote,urlsplit
from download_verified import file_digest


def audit(lock_path,root):
    lock_path=Path(lock_path)
    lock=tomllib.loads(lock_path.read_text())
    if lock.get('lock-version')!='1.0':
        raise ValueError('Unsupported lock format')
    root=Path(root)
    packages=[]
    for package in lock['packages']:
        entry=dict(name=package['name'],version=package['version'])
        wheels=package.get('wheels',[])
        if not wheels:
            entry['state']='source_build_required'
        else:
            wheel=wheels[0]
            filename=unquote(urlsplit(wheel['url']).path.rsplit('/',1)[-1])
            if Path(filename).name!=filename or not filename.endswith('.whl'):
                raise ValueError('Unsafe wheel filename')
            path=root/filename
            entry['file']=filename
            entry['expected_sha256']=wheel['hashes']['sha256']
            if path.is_file():
                entry['actual_sha256']=file_digest(path)
                entry['bytes']=path.stat().st_size
                entry['state']='verified' if entry['actual_sha256']==entry['expected_sha256'] else 'mismatch'
            else:
                partial=path.with_name(path.name+'.part')
                entry['state']='partial' if partial.is_file() else 'missing'
                entry['bytes']=partial.stat().st_size if partial.is_file() else 0
        packages.append(entry)
    counts={state:sum(p['state']==state for p in packages) for state in sorted({p['state'] for p in packages})}
    return dict(lock_sha256=file_digest(lock_path),package_count=len(packages),counts=counts,
                ready=bool(packages) and all(p['state']=='verified' for p in packages),packages=packages)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('lock')
    parser.add_argument('--root',default='.wheelhouse')
    parser.add_argument('--summary',action='store_true')
    args=parser.parse_args()
    report=audit(args.lock,args.root)
    if args.summary:
        report.pop('packages')
    print(json.dumps(report,indent=2))
    if not report['ready']:
        raise SystemExit(1)


if __name__=='__main__':
    main()
