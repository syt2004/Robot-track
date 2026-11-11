"""Materialize single-platform uv pylock files with resumable verified downloads.

Requires Python >=3.11. Does not install anything or execute downloaded wheels.
All wheels are retained for offline reproduction; no missing package is skipped.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import tomllib
from urllib.parse import urlsplit, unquote
from download_verified import download
from download_parallel import download_parallel
from download_verified import file_digest


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('locks',nargs='+')
    parser.add_argument('--output',default='.wheelhouse')
    parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--exclude-lock',action='append',default=[],
                        help='Do not transfer identical wheels already owned by another active lock download')
    parser.add_argument('--range-workers',type=int,default=1,
                        help='Concurrent bounded ranges per wheel; existing partial files are reused')
    args=parser.parse_args()
    if not 1<=args.workers<=8:
        parser.error('workers must be between 1 and 8')
    if not 1<=args.range_workers<=16:
        parser.error('range-workers must be between 1 and 16')
    root=Path(args.output).resolve()
    root.mkdir(parents=True,exist_ok=True)
    downloads={}
    for lock_path in map(Path,args.locks):
        lock=tomllib.loads(lock_path.read_text())
        if lock.get('lock-version')!='1.0':
            raise ValueError('Unsupported lock format')
        lines=[]
        for package in lock['packages']:
            wheels=package.get('wheels',[])
            if not wheels:
                raise ValueError('A prebuilt wheel is required for every package')
            # uv can retain multiple build revisions for the same target tags.
            # Select its first listed wheel and pin that exact hash below.
            wheel=wheels[0]
            url,digest=wheel['url'],wheel['hashes']['sha256']
            filename=unquote(urlsplit(url).path.rsplit('/',1)[-1])
            if Path(filename).name!=filename or not filename.endswith('.whl'):
                raise ValueError('Unsafe wheel filename')
            previous=downloads.get(filename)
            if previous and previous[1]!=digest:
                raise ValueError('Conflicting wheel hashes')
            downloads[filename]=(url,digest)
            lines.append('%s==%s --hash=sha256:%s' % (package['name'],package['version'],digest))
        # Use these pins with --no-index --find-links and --require-hashes.
        lock_path.with_suffix('.offline.txt').write_text('\n'.join(lines)+'\n')
    excluded={}
    for lock_path in map(Path,args.exclude_lock):
        other=tomllib.loads(lock_path.read_text())
        for package in other['packages']:
            wheel=package['wheels'][0]
            name=unquote(urlsplit(wheel['url']).path.rsplit('/',1)[-1])
            excluded[name]=wheel['hashes']['sha256']
    for name in list(downloads):
        if name in excluded:
            if downloads[name][1]!=excluded[name]:
                raise ValueError('Excluded wheel has conflicting digest: '+name)
            del downloads[name]
    failures=[]
    def transfer(url,path,digest):
        # uv records already-resolved local wheels as file:// URLs. Never send
        # these through HTTP, and never treat a local filename as verification.
        if urlsplit(url).scheme=='file':
            if not path.is_file() or file_digest(path)!=digest:
                raise ValueError('Locked local wheel missing or mismatched in output: '+path.name)
            return path
        if args.range_workers==1:
            return download(url,path,digest)
        return download_parallel(url,path,digest,workers=args.range_workers)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures={pool.submit(transfer,url,root/name,digest):name
                 for name,(url,digest) in downloads.items()}
        for future in as_completed(futures):
            name=futures[future]
            try:
                future.result()
                print('VERIFIED '+name,flush=True)
            except Exception as error:
                failures.append(name)
                print('INCOMPLETE %s: %s' % (name,error),flush=True)
    if failures:
        raise SystemExit('%d wheels remain incomplete; rerun to resume' % len(failures))
    if args.exclude_lock:
        print('Selected transfers complete; excluded locks must finish before installation',flush=True)


if __name__=='__main__':
    main()
