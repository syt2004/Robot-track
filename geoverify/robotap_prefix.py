"""Validate an official ZIP member from a bounded archive prefix, not a full ZIP.

The member index is obtained from the pinned GCS object central directory.
CRC32 checks transfer integrity; local SHA256 records reproducibility. Neither
is reported as verification of the unavailable whole-archive MD5.
"""
import json
from pathlib import Path
import struct
import zlib
from .datasets import file_sha256
from .geometry_worker import atomic_json


def member_header(stream, entry):
    stream.seek(entry['offset'])
    header=stream.read(30)
    if len(header)!=30:raise ValueError('Truncated ZIP member header')
    sig,version,flags,method,mtime,mdate,crc,compressed,uncompressed,nlen,xlen=struct.unpack('<4s5H3I2H',header)
    name=stream.read(nlen);extra=stream.read(xlen)
    if sig!=b'PK\x03\x04' or flags!=0 or method!=8 or name.decode('utf8')!=entry['name'] or len(extra)!=xlen:
        raise ValueError('ZIP member identity/method mismatch')
    if compressed==0xffffffff or uncompressed==0xffffffff:
        cursor=0;zip64=None
        while cursor+4<=len(extra):
            kind,size=struct.unpack_from('<HH',extra,cursor);cursor+=4
            if cursor+size>len(extra):raise ValueError('Malformed ZIP extra field')
            if kind==1:zip64=extra[cursor:cursor+size]
            cursor+=size
        if zip64 is None:raise ValueError('Missing ZIP64 sizes')
        offset=0
        if uncompressed==0xffffffff:
            uncompressed=struct.unpack_from('<Q',zip64,offset)[0];offset+=8
        if compressed==0xffffffff:compressed=struct.unpack_from('<Q',zip64,offset)[0]
    if (crc,compressed,uncompressed)!=(entry['crc'],entry['compressed'],entry['uncompressed']):
        raise ValueError('ZIP header disagrees with pinned central directory')
    return stream.tell()


def extract_member(prefix, entry, output):
    prefix,output=Path(prefix),Path(output)
    output.parent.mkdir(parents=True,exist_ok=True)
    pending=output.with_name(output.name+'.pending')
    crc=0;size=0
    with prefix.open('rb') as source,pending.open('wb') as target:
        data_start=member_header(source,entry)
        if prefix.stat().st_size<data_start+entry['compressed']:
            raise ValueError('Incomplete archive prefix')
        decoder=zlib.decompressobj(-15)
        remaining=entry['compressed']
        while remaining:
            data=source.read(min(1024*1024,remaining))
            if not data:raise ValueError('Truncated compressed member')
            remaining-=len(data)
            while data:
                block=decoder.decompress(data,8*1024*1024)
                if decoder.unused_data:raise ValueError('Unexpected data after compressed member')
                data=decoder.unconsumed_tail
                size+=len(block)
                if size>entry['uncompressed']:raise ValueError('Member exceeds declared size')
                crc=zlib.crc32(block,crc);target.write(block)
        if not decoder.eof or decoder.unused_data or size!=entry['uncompressed'] or crc!=entry['crc']:
            raise ValueError('ZIP member length/CRC validation failed')
    digest=file_sha256(pending)
    if output.exists():
        if file_sha256(output)!=digest:raise ValueError('Existing extracted member differs')
        pending.unlink()
    else:pending.replace(output)
    return dict(member=entry['name'],bytes=size,crc32=crc,sha256=digest,data_offset=data_start,
                compressed_bytes=entry['compressed'],whole_archive_md5_verified=False)


def write_provenance(prefix,index_path,member_path,entry):
    record=extract_member(prefix,entry,member_path)
    record.update(source_index_sha256=file_sha256(index_path),
                  source=json.loads(Path(index_path).read_text())['source'],
                  prefix_sha256=file_sha256(prefix),prefix_bytes=Path(prefix).stat().st_size)
    atomic_json(Path(member_path).with_suffix('.provenance.json'),record)
    return record
