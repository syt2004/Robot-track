"""Fetch the complete pinned CoTracker3 Kubric archive inventory, not a subset."""
import argparse
import fcntl
import json
from pathlib import Path
import re
import time
from urllib.parse import urlsplit
from urllib.request import Request,urlopen

REPOSITORY='facebook/CoTracker3_Kubric'
REVISION='a23b6e6e332595c9d92944680a10415e93bfd1e1'
ARCHIVE_COUNT=5869
TREE='https://huggingface.co/api/datasets/'+REPOSITORY+'/tree/'+REVISION


def next_link(header):
    links=re.findall(r'<([^>]+)>;\s*rel="next"',header or '')
    if len(links)>1:
        raise ValueError('Ambiguous pagination')
    if not links:
        return None
    link=links[0]
    parsed=urlsplit(link)
    if parsed.scheme!='https' or parsed.netloc!='huggingface.co' or parsed.path!=urlsplit(TREE).path:
        raise ValueError('Pagination left pinned official repository')
    return link


def parse_archives(items):
    result=[]
    for entry in items:
        path=entry['path']
        if not path.endswith('.tar.gz'):
            continue
        if not re.fullmatch(r'\d{4}\.tar\.gz',path) or entry.get('type')!='file':
            raise ValueError('Unexpected archive name/type')
        digest=entry.get('lfs',{}).get('oid','')
        size=entry.get('size')
        if not re.fullmatch('[0-9a-f]{64}',digest) or not isinstance(size,int) or size<=0:
            raise ValueError('Archive missing official size/SHA256')
        if entry['lfs'].get('size')!=size:
            raise ValueError('LFS size mismatch')
        result.append(dict(path=path,bytes=size,sha256=digest,
                           url='https://huggingface.co/datasets/'+REPOSITORY+'/resolve/'+REVISION+'/'+path))
    return result


def fetch_page(url):
    for attempt in range(5):
        try:
            with urlopen(Request(url,headers={'User-Agent':'GeoVerifyTrack/1.0'}),timeout=40) as response:
                data=json.load(response)
                if not isinstance(data,list):
                    raise ValueError('Expected dataset tree list')
                return parse_archives(data),next_link(response.headers.get('Link'))
        except Exception as error:
            if attempt==4:
                raise RuntimeError('Inventory page failed: '+type(error).__name__) from None
            time.sleep(min(2**attempt,8))


def atomic_json(path,data):
    pending=path.with_suffix(path.suffix+'.pending')
    pending.write_text(json.dumps(data,indent=2)+'\n')
    pending.replace(path)


def inventory(output):
    output=Path(output)
    output.parent.mkdir(parents=True,exist_ok=True)
    progress=output.with_suffix(output.suffix+'.progress')
    with output.with_suffix(output.suffix+'.lock').open('a') as lock:
        fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        if output.exists():
            raise FileExistsError('Inventory already exists; do not overwrite locked metadata')
        state=dict(repository=REPOSITORY,revision=REVISION,files=[],
                   next_url=TREE+'?recursive=false&limit=1000',visited=[])
        if progress.exists():
            state=json.loads(progress.read_text())
            if state['repository']!=REPOSITORY or state['revision']!=REVISION:
                raise ValueError('Saved inventory revision mismatch')
        names={entry['path'] for entry in state['files']}
        if len(names)!=len(state['files']):
            raise ValueError('Duplicate saved archive')
        while state['next_url']:
            url=state['next_url']
            # Validate persisted pagination as well as remote response links.
            next_link('<'+url+'>; rel="next"')
            if url in state['visited']:
                raise ValueError('Pagination cycle')
            entries,next_url=fetch_page(url)
            for entry in entries:
                if entry['path'] in names:
                    raise ValueError('Duplicate archive across pages')
                names.add(entry['path'])
            state['files'].extend(entries)
            state['visited'].append(url)
            state['next_url']=next_url
            atomic_json(progress,state)
            print('Inventoried %d/%d archives'%(len(names),ARCHIVE_COUNT),flush=True)
        expected={'%04d.tar.gz'%i for i in range(ARCHIVE_COUNT)}
        if names!=expected:
            raise ValueError('Incomplete archive identity set; no final manifest published')
        result=dict(schema_version=1,repository=REPOSITORY,revision=REVISION,
                    archive_count=len(names),total_bytes=sum(f['bytes'] for f in state['files']),
                    files=sorted(state['files'],key=lambda f:f['path']))
        atomic_json(output,result)
        print('Complete inventory: %d bytes in %d archives'%(result['total_bytes'],result['archive_count']),flush=True)
        return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',default='configs/kubric.archives.lock.json')
    inventory(parser.parse_args().output)
