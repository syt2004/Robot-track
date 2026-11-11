"""Fetch pinned official source without large presentation media.

Every downloaded blob is verified against the official commit's Git tree.
Submodules and skipped presentation files are recorded, not claimed installed.
"""
import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path, PurePosixPath
import subprocess

COMMIT = 'fbd59ffadf2de9fccba5ea017e13af239075fbe9'
REPO = 'TencentARC/Track4World'


def fetch(url):
    result = subprocess.run(['curl','--fail','--location','--silent','--show-error',
                             '--connect-timeout','10','--max-time','45','--retry','2', url],
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode:
        raise RuntimeError('Source download failed (curl %d)' % result.returncode)
    return result.stdout


def git_hash(data):
    return hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='third_party/track4world')
    parser.add_argument('--repo', default=REPO)
    parser.add_argument('--commit', default=COMMIT)
    parser.add_argument('--include-prefix', default='')
    args = parser.parse_args()
    repo, commit = args.repo, args.commit
    import re
    if not re.fullmatch(r'[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+',repo) or not re.fullmatch(r'[0-9a-f]{40}',commit):
        raise ValueError('Expected GitHub owner/repository and full commit hash')
    root = Path(args.output).resolve()
    root.mkdir(parents=True, exist_ok=True)
    tree_path = root / 'SOURCE_TREE.json'
    if tree_path.exists():
        tree = json.loads(tree_path.read_text())
    else:
        tree = json.loads(fetch('https://api.github.com/repos/'+repo+'/git/trees/'+commit+'?recursive=1'))
    if tree.get('sha') != commit or tree.get('truncated', True):
        raise ValueError('Incomplete or incorrect official source tree')
    tree_path.write_text(json.dumps(tree, indent=2))
    entries, skipped, submodules = [], [], []
    for entry in tree['tree']:
        path = PurePosixPath(entry['path'])
        if path.is_absolute() or '..' in path.parts:
            raise ValueError('Unsafe source path')
        if entry['type'] == 'commit':
            submodules.append(entry)
        elif entry['type'] == 'blob':
            if (path.parts[0] in ('assets','demo_data') or
                (args.include_prefix and not entry['path'].startswith(args.include_prefix)
                 and path.name not in ('README.md','LICENSE','LICENSE.txt','LICENSE.md'))):
                skipped.append(entry['path'])
            elif entry['mode'] not in ('100644','100755'):
                raise ValueError('Unsupported source file mode')
            else:
                entries.append(entry)
    # Read the model/demo/config first even if a later auxiliary download fails.
    entries.sort(key=lambda e: (e['path'] not in ('README.md','demo.py','track4world/nets/model.py'), e['path']))

    def one(entry):
        destination = root / entry['path']
        if destination.exists():
            if git_hash(destination.read_bytes()) == entry['sha']:
                return
            raise ValueError('Existing source differs: '+entry['path'])
        try:
            data = fetch('https://raw.githubusercontent.com/'+repo+'/'+commit+'/'+entry['path'])
        except RuntimeError:
            blob = json.loads(fetch(entry['url']))
            if blob.get('encoding') != 'base64':
                raise ValueError('Unexpected Git blob encoding')
            data = base64.b64decode(blob['content'])
        if git_hash(data) != entry['sha']:
            raise ValueError('Source hash mismatch: '+entry['path'])
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(data)
        print('Verified '+entry['path'], flush=True)

    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(one, entries))
    (root/'SOURCE_MANIFEST.json').write_text(json.dumps(dict(
        repository=repo, commit=commit, verified_blobs=len(entries),
        skipped_presentation_files=skipped, uninstalled_submodules=submodules), indent=2))


if __name__ == '__main__':
    main()
