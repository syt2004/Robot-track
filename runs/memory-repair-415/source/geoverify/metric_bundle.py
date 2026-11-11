"""Identity checks for the unchanged DA3 metric tensors fetched by byte range."""
import hashlib
import json
from pathlib import Path
import struct
from .datasets import file_sha256


def compact_header(source_header,lock):
    raw=source_header[8:]
    if struct.unpack('<Q',source_header[:8])[0]!=len(raw) or hashlib.sha256(source_header).hexdigest()!=lock['source_header_sha256']:
        raise ValueError('Source safetensors header mismatch')
    header=json.loads(raw)
    selected={k:dict(v) for k,v in header.items() if k.startswith(lock['prefix'])}
    if len(selected)!=lock['tensor_count']:raise ValueError('Metric tensor count mismatch')
    start=lock['range_start']-len(source_header);end=lock['range_end_exclusive']-len(source_header)
    ordered=sorted(selected,key=lambda k:selected[k]['data_offsets'][0]);cursor=start
    for key in ordered:
        a,b=selected[key]['data_offsets']
        if a!=cursor or b<=a:raise ValueError('Metric data must be contiguous')
        selected[key]['data_offsets']=[a-start,b-start];cursor=b
    if cursor!=end or end-start!=lock['payload_bytes']:raise ValueError('Metric payload range mismatch')
    raw=json.dumps(selected,sort_keys=True,separators=(',',':')).encode()
    raw+=b' '*((-len(raw))%8)
    return struct.pack('<Q',len(raw))+raw


def audit_bundle(directory,lock):
    directory=Path(directory)
    provenance=json.loads((directory/'metric_source.json').read_text())
    if provenance['selection']!=lock:raise ValueError('Metric source selection mismatch')
    for name in ('model.safetensors','config.json','source-header.bin'):
        if file_sha256(directory/name)!=provenance['files'][name]:raise ValueError('Metric bundle checksum mismatch: '+name)
    expected=compact_header((directory/'source-header.bin').read_bytes(),lock)
    with (directory/'model.safetensors').open('rb') as stream:
        if stream.read(len(expected))!=expected:raise ValueError('Repacked metric tensor schema mismatch')
    if (directory/'model.safetensors').stat().st_size!=len(expected)+lock['payload_bytes']:
        raise ValueError('Metric bundle length mismatch')
    return provenance
