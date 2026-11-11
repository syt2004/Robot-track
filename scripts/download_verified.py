"""Resumable bounded-range HTTPS download; publish only after SHA256 matches.

Uses no credentials and logs no redirected/signed URLs. A .part file is never
considered a model. Servers must honor ranges for files larger than one chunk.
"""
import argparse
import hashlib
import os
from pathlib import Path
import re
import time
from urllib.request import Request, urlopen


def file_digest(path, algorithm='sha256'):
    result = hashlib.new(algorithm)
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            result.update(block)
    return result.hexdigest()


def sha256(path):
    return file_digest(path)


def download(url, output, expected_sha256, chunk_bytes=512*1024, retries=5, algorithm='sha256'):
    if not url.startswith('https://'):
        raise ValueError('An HTTPS source is required')
    if algorithm not in ('sha256','md5'):
        raise ValueError('Unsupported integrity algorithm')
    length = 64 if algorithm=='sha256' else 32
    if not re.fullmatch('[0-9a-f]{%d}' % length, expected_sha256):
        raise ValueError('A trusted source digest is required')
    if chunk_bytes < 1 or retries < 1:
        raise ValueError('Positive chunk size and retries required')
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        if file_digest(output,algorithm) != expected_sha256:
            raise ValueError('Existing destination hash mismatch; refusing overwrite')
        return output
    part = output.with_name(output.name + '.part')
    # Exclusive file lock avoids two processes appending to one partial file.
    import fcntl
    with part.open('ab+') as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        offset = stream.tell()
        if offset and file_digest(part,algorithm) == expected_sha256:
            os.replace(part, output)
            return output
        total = None
        while total is None or offset < total:
            for attempt in range(retries):
                try:
                    request = Request(url, headers={
                        'Range': 'bytes=%d-%d' % (offset, offset+chunk_bytes-1),
                        'Accept-Encoding': 'identity', 'User-Agent': 'GeoVerifyTrack/1.0'})
                    with urlopen(request, timeout=40) as response:
                        if response.status == 206:
                            match = re.fullmatch(r'bytes (\d+)-(\d+)/(\d+)',
                                                 response.headers.get('Content-Range', ''))
                            if not match:
                                raise ValueError('Invalid Content-Range')
                            start, end, observed_total = map(int, match.groups())
                            if start != offset or end < start or end >= observed_total:
                                raise ValueError('Wrong response byte interval')
                            length = end-start+1
                            if length > chunk_bytes:
                                raise ValueError('Response exceeds requested interval')
                        elif response.status == 200 and offset == 0:
                            length = int(response.headers.get('Content-Length', '0'))
                            if not 0 < length <= chunk_bytes:
                                raise ValueError('Server does not support bounded ranges')
                            observed_total = length
                        else:
                            raise ValueError('Server did not honor resume request')
                        if total is not None and observed_total != total:
                            raise ValueError('Remote size changed during transfer')
                        data = response.read(length+1)
                        if len(data) != length:
                            raise ValueError('Truncated or oversized chunk')
                    break
                except Exception as error:
                    if attempt == retries-1:
                        raise RuntimeError('Download failed at byte %d (%s)' %
                                           (offset, type(error).__name__)) from None
                    print('%s retry byte %d: %s' % (output.name, offset, type(error).__name__), flush=True)
                    time.sleep(min(2**attempt, 8))
            # Never append incomplete chunks; retry starts at the same offset.
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
            offset += length
            total = observed_total
            print('%s %d/%d bytes' % (output.name, offset, total), flush=True)
        if file_digest(part,algorithm) != expected_sha256:
            raise ValueError('Downloaded digest mismatch; retained .part, not published')
        os.replace(part, output)
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', required=True)
    parser.add_argument('--output', required=True)
    hashes=parser.add_mutually_exclusive_group(required=True)
    hashes.add_argument('--sha256')
    hashes.add_argument('--md5',help='For official sources publishing MD5 only; not a signature')
    args = parser.parse_args()
    path=download(args.url,args.output,args.sha256 or args.md5,
                  algorithm='sha256' if args.sha256 else 'md5')
    print(path)
    print('SHA256 '+sha256(path))
