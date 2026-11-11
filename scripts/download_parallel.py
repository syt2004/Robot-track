"""Bounded parallel ranges, ordered append, existing sequential .part compatible.

Only contiguous, completely received chunks enter .part. No sparse-file holes,
sidecar completion claims or unverified final files. HTTPS and final trusted
digest checks remain mandatory. Logs never contain redirected signed URLs.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import os
from pathlib import Path
import re
import time
from urllib.request import Request,urlopen
try:
    from scripts.download_verified import file_digest
except ModuleNotFoundError:
    from download_verified import file_digest


def fetch_range(url,start,end,total=None,retries=5):
    for attempt in range(retries):
        try:
            request=Request(url,headers={'Range':'bytes=%d-%d'%(start,end),
                                          'Accept-Encoding':'identity','User-Agent':'GeoVerifyTrack/1.0'})
            with urlopen(request,timeout=40) as response:
                interval=re.fullmatch(r'bytes (\d+)-(\d+)/(\d+)',response.headers.get('Content-Range',''))
                if response.status!=206 or interval is None:
                    raise ValueError('Server must honor bounded byte ranges')
                actual_start,actual_end,size=map(int,interval.groups())
                if actual_start!=start or actual_end!=min(end,size-1) or size<=start:
                    raise ValueError('Wrong response interval')
                if total is not None and size!=total:
                    raise ValueError('Remote size changed')
                length=actual_end-start+1
                data=response.read(length+1)
                if len(data)!=length:
                    raise ValueError('Incomplete or oversized response')
                return data,size
        except Exception as error:
            if attempt==retries-1:
                # Only our fixed validation messages are safe to report.
                # Network exception strings can contain signed redirect URLs.
                known=('Server must honor bounded byte ranges','Wrong response interval',
                       'Remote size changed','Incomplete or oversized response')
                reason=str(error) if isinstance(error,ValueError) and str(error) in known else type(error).__name__
                raise RuntimeError('Range failed at byte %d (%s)'%(start,reason)) from None
            time.sleep(min(2**attempt,8))


def download_parallel(url,output,digest,algorithm='sha256',workers=4,chunk_bytes=512*1024,retries=5):
    if not url.startswith('https://') or algorithm not in ('sha256','md5'):
        raise ValueError('HTTPS and supported digest required')
    if not re.fullmatch('[0-9a-f]{%d}'%(64 if algorithm=='sha256' else 32),digest):
        raise ValueError('Trusted source digest required')
    if not 1<=workers<=64 or not 1<=chunk_bytes<=16*1024*1024 or retries<1:
        raise ValueError('Invalid concurrency, chunk size or retries')
    output=Path(output)
    output.parent.mkdir(parents=True,exist_ok=True)
    if output.exists():
        if file_digest(output,algorithm)!=digest:
            raise ValueError('Existing destination hash mismatch; refusing overwrite')
        return output
    part=output.with_name(output.name+'.part')
    with part.open('ab+') as stream:
        fcntl.flock(stream.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        offset=stream.tell()
        if offset and file_digest(part,algorithm)==digest:
            os.replace(part,output)
            return output
        # Discover total without trusting HEAD or a signed redirect's metadata.
        _,total=fetch_range(url,0,0,retries=retries)
        if offset>=total:
            raise ValueError('Partial file is full/oversized but has wrong digest; retained for diagnosis')
        with ThreadPoolExecutor(max_workers=workers) as pool:
            pending={}
            next_start=offset
            def fill():
                nonlocal next_start
                while len(pending)<workers and next_start<total:
                    end=min(next_start+chunk_bytes,total)-1
                    pending[next_start]=pool.submit(fetch_range,url,next_start,end,total,retries)
                    next_start=end+1
            fill()
            while pending:
                data,_=pending.pop(offset).result()
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
                offset+=len(data)
                print('%s %d/%d bytes (%d workers)'%(output.name,offset,total,workers),flush=True)
                fill()
        if file_digest(part,algorithm)!=digest:
            raise ValueError('Downloaded digest mismatch; retained .part, not published')
        os.replace(part,output)
    return output


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--chunk-bytes',type=int,default=512*1024,
                        help='Range size, 1..16777216 bytes; resume preserves the contiguous prefix')
    hashes=parser.add_mutually_exclusive_group(required=True)
    hashes.add_argument('--sha256')
    hashes.add_argument('--md5')
    args=parser.parse_args()
    path=download_parallel(args.url,args.output,args.sha256 or args.md5,
                           'sha256' if args.sha256 else 'md5',args.workers,chunk_bytes=args.chunk_bytes)
    print(str(path)+' SHA256 '+file_digest(path),flush=True)


if __name__=='__main__':
    main()
