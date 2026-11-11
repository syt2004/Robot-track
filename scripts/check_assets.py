"""Read-only audit of actual files against the pinned asset lock; no downloads."""
import argparse
import hashlib
import json
from pathlib import Path


def check_asset(root,entry):
    path=Path(root)/entry['path']
    if not path.is_file():
        part=path.with_name(path.name+'.part')
        return dict(status='partial' if part.is_file() else 'missing',
                    actual_bytes=part.stat().st_size if part.is_file() else 0,
                    expected_bytes=entry['bytes'])
    size=path.stat().st_size
    if size!=entry['bytes']:
        return dict(status='mismatch',reason='size',actual_bytes=size,expected_bytes=entry['bytes'])
    if 'sha256' in entry:
        algorithm,expected='sha256',entry['sha256']
    elif 'md5' in entry:
        algorithm,expected='md5',entry['md5']
    elif 'git_blob_sha1' in entry:
        algorithm,expected='git_blob_sha1',entry['git_blob_sha1']
    else:
        raise ValueError('No pinned digest: '+entry['path'])
    digest=hashlib.new('sha1' if algorithm=='git_blob_sha1' else algorithm)
    if algorithm=='git_blob_sha1':
        digest.update(b'blob '+str(size).encode()+b'\0')
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):
            digest.update(block)
    actual=digest.hexdigest()
    return dict(status='verified' if actual==expected else 'mismatch',
                algorithm=algorithm,digest=actual,actual_bytes=size,expected_bytes=entry['bytes'])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',default='.')
    parser.add_argument('--lock',default='configs/assets.lock.json')
    args=parser.parse_args()
    lock=json.loads(Path(args.lock).read_text())
    results={name:check_asset(args.root,entry) for name,entry in lock['downloads'].items()}
    print(json.dumps(dict(assets=results,not_yet_locked=lock.get('not_yet_locked',[])),indent=2))
    if any(r['status']!='verified' for r in results.values()) or lock.get('not_yet_locked'):
        raise SystemExit(1)


if __name__=='__main__':
    main()
