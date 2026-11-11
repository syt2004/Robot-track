"""Download every archive in the explicitly selected, checksum-locked subset."""
import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
import json
from pathlib import Path
import re
try:
    from scripts.download_parallel import download_parallel
except ModuleNotFoundError:
    from download_parallel import download_parallel


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',default='configs/kubric.small.lock.json')
    parser.add_argument('--output',default='data/downloads/kubric')
    parser.add_argument('--file-workers',type=int,default=2)
    parser.add_argument('--range-workers',type=int,default=2)
    parser.add_argument('--chunk-bytes',type=int,default=512*1024)
    args=parser.parse_args()
    if not 1<=args.file_workers<=4 or not 1<=args.range_workers<=8:
        parser.error('file-workers must be 1..4 and range-workers 1..8')
    if not 1<=args.chunk_bytes<=16*1024*1024:
        parser.error('chunk-bytes must be 1..16777216')
    manifest=json.loads(Path(args.manifest).read_text())
    files=manifest['files']
    if manifest['repository']!='facebook/CoTracker3_Kubric' or not re.fullmatch('[0-9a-f]{40}',manifest['revision']):
        raise ValueError('Expected pinned official Kubric source')
    if len(files)!=manifest['selected_archive_count'] or len({f['path'] for f in files})!=len(files):
        raise ValueError('Incomplete/duplicate selected archive list')
    if sum(f['bytes'] for f in files)!=manifest['total_compressed_bytes']:
        raise ValueError('Selected total size mismatch')
    for entry in files:
        if not re.fullmatch(r'\d{4}\.tar\.gz',entry['path']) or entry['split'] not in ('train','val'):
            raise ValueError('Invalid archive path/split')
        expected='https://huggingface.co/datasets/'+manifest['repository']+'/resolve/'+manifest['revision']+'/'+entry['path']
        if entry['url']!=expected:
            raise ValueError('Archive URL does not match pinned identity')
    root=Path(args.output)
    root.mkdir(parents=True,exist_ok=True)
    failures=[]
    def transfer(entry):
        result=download_parallel(entry['url'],root/entry['path'],entry['sha256'],workers=args.range_workers,
                                 chunk_bytes=args.chunk_bytes)
        if result.stat().st_size!=entry['bytes']:
            raise ValueError('Verified archive differs from inventory size')
        return result
    with ThreadPoolExecutor(max_workers=args.file_workers) as pool:
        futures={pool.submit(transfer,entry):entry['path'] for entry in files}
        for future in as_completed(futures):
            name=futures[future]
            try:
                future.result()
                print('VERIFIED '+name,flush=True)
            except Exception as error:
                failures.append(name)
                print('INCOMPLETE '+name+': '+str(error),flush=True)
    if failures:
        raise SystemExit('%d selected archives incomplete; rerun to resume'%len(failures))
    print('Verified all %d selected archives (not the full source dataset)'%len(files),flush=True)


if __name__=='__main__':
    main()
