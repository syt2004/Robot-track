"""Fetch only the exact pretrained DA3 metric branch used after Track4World load."""
from concurrent.futures import ThreadPoolExecutor
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.download_parallel import fetch_range
from scripts.check_assets import check_asset
from geoverify.datasets import file_sha256
from geoverify.geometry_worker import atomic_json
from geoverify.metric_bundle import compact_header,audit_bundle


def main():
    root=Path(__file__).resolve().parents[1]
    lock=json.loads((root/'configs/da3.metric.lock.json').read_text())
    out=root/'weights/da3nested-metric';out.mkdir(parents=True,exist_ok=True)
    if (out/'metric_source.json').exists():
        audit_bundle(out,lock);print('Verified existing metric bundle');return
    assets=json.loads((root/'configs/assets.lock.json').read_text())['downloads']
    if lock['source_sha256']!=assets['da3_nested']['sha256'] or check_asset(root,assets['da3_config'])['status']!='verified':
        raise ValueError('Official source identity/config mismatch')
    # Authenticate the recorded header through the same pinned HTTPS source.
    source_header,_=fetch_range(lock['source_url'],0,lock['source_header_bytes']-1,lock['source_bytes'])
    header=compact_header(source_header,lock)
    (out/'source-header.bin').write_bytes(source_header)
    part=out/'metric.payload.part'
    with part.open('ab+') as stream:
        fcntl.flock(stream.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        offset=stream.tell();end=lock['payload_bytes']
        if offset>end:raise ValueError('Metric partial file oversized')
        with ThreadPoolExecutor(max_workers=16) as pool:
            pending={};next_start=offset
            def fill():
                nonlocal next_start
                while len(pending)<16 and next_start<end:
                    last=min(next_start+1048576,end)-1
                    pending[next_start]=pool.submit(fetch_range,lock['source_url'],lock['range_start']+next_start,
                                                   lock['range_start']+last,lock['source_bytes'])
                    next_start=last+1
            fill()
            while pending:
                data,_=pending.pop(offset).result();stream.write(data);stream.flush();os.fsync(stream.fileno())
                offset+=len(data);print('Metric tensors',offset,'/',end,flush=True);fill()
        packed=out/'model.safetensors.pending'
        with packed.open('wb') as destination,part.open('rb') as source:
            destination.write(header);shutil.copyfileobj(source,destination,1048576)
        packed.replace(out/'model.safetensors')
        shutil.copyfile(root/assets['da3_config']['path'],out/'config.json')
        provenance=dict(selection=lock,whole_source_sha256_verified_locally=False,
                        files={name:file_sha256(out/name) for name in ('model.safetensors','config.json','source-header.bin')},
                        note='Exact metric tensor bytes from pinned HTTPS source; anyview branch must be completely loaded from official Track4World checkpoint before inference')
        atomic_json(out/'metric_source.json',provenance);audit_bundle(out,lock)
        part.replace(out/'metric.payload')
    print('Verified selected tensor bundle schema and recorded local hashes',flush=True)


if __name__=='__main__':main()
