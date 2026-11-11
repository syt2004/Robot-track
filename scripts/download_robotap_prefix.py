"""Resume only the prefix containing RoboTAP split4; verify and extract that member."""
from concurrent.futures import ThreadPoolExecutor
import argparse
import fcntl
import json
import os
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.download_parallel import fetch_range
from geoverify.robotap_prefix import member_header,write_provenance


def main(root=None,workers=4):
    if not isinstance(workers,int) or not 1<=workers<=16:
        raise ValueError('RoboTAP workers must be between 1 and 16')
    root=Path(__file__).resolve().parents[1] if root is None else Path(root)
    index_path=root/'configs/robotap.zip-index.json'
    index=json.loads(index_path.read_text())
    entry=next(e for e in index['entries'] if e['name']=='robotap/robotap_split4.pkl')
    destination=root/'data/downloads/robotap-split4-prefix.bin'
    part=destination.with_name(destination.name+'.part')
    member=root/'data/downloads/robotap_split4.pkl'
    destination.parent.mkdir(parents=True,exist_ok=True)
    if destination.exists():
        print(write_provenance(destination,index_path,member,entry));return
    with part.open('ab+') as stream:
        fcntl.flock(stream.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        stream.seek(0,2);offset=stream.tell()
        if offset<entry['offset']+4096:
            data,_=fetch_range(index['source'],0,entry['offset']+4095,index['bytes'])
            stream.seek(0,2)
            stream.write(data[offset:]);stream.flush();offset=stream.tell()
        end=member_header(stream,entry)+entry['compressed']
        stream.seek(0,2)
        if offset>end:raise ValueError('Prefix exceeds requested member boundary')
        with ThreadPoolExecutor(max_workers=workers) as pool:
            pending={};next_start=offset
            def fill():
                nonlocal next_start
                while len(pending)<workers and next_start<end:
                    last=min(next_start+1048576,end)-1
                    pending[next_start]=pool.submit(fetch_range,index['source'],next_start,last,index['bytes'])
                    next_start=last+1
            fill()
            while pending:
                data,_=pending.pop(offset).result();stream.write(data);stream.flush()
                os.fsync(stream.fileno())
                offset+=len(data);print('split4 prefix',offset,'/',end,flush=True);fill()
        print(write_provenance(part,index_path,member,entry),flush=True)
        part.replace(destination)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers',type=int,default=4)
    args=parser.parse_args()
    main(workers=args.workers)
