"""Deterministic full-video subset from verified RoboTAP split4 only.

This is a bandwidth-limited subset of the upstream evaluation shards, not the
previous split3+split4 selection and not the complete RoboTAP benchmark.
"""
import argparse
import hashlib
import json
from pathlib import Path
import pickle
import numpy as np
from .datasets import NumpyDatasetUnpickler,file_sha256,iter_robotap
from .geometry_worker import atomic_json


def prepare(member,index_path,output,fps,fps_description,videos=3,seed=20261029):
    if videos<1 or not np.isfinite(fps) or fps<=0 or not fps_description.strip():
        raise ValueError('Positive video count/FPS and explicit FPS provenance required')
    member,index_path,output=Path(member).resolve(),Path(index_path).resolve(),Path(output).resolve()
    if output.exists():raise FileExistsError(output)
    provenance=json.loads(member.with_suffix('.provenance.json').read_text())
    index=json.loads(index_path.read_text())
    entry=next(e for e in index['entries'] if e['name']=='robotap/robotap_split4.pkl')
    if (provenance['source_index_sha256']!=file_sha256(index_path) or provenance['source']!=index['source']
        or provenance['member']!=entry['name'] or provenance['crc32']!=entry['crc']
        or provenance['bytes']!=entry['uncompressed'] or member.stat().st_size!=entry['uncompressed']
        or provenance['sha256']!=file_sha256(member)):
        raise ValueError('Member provenance/checksum mismatch')
    with member.open('rb') as stream:
        records=NumpyDatasetUnpickler(stream).load()
        if stream.read(1):raise ValueError('Unexpected trailing member bytes')
    if not isinstance(records,dict) or not records or any(not isinstance(k,str) for k in records):
        raise ValueError('Expected named RoboTAP records')
    if len(records)<videos:raise ValueError('Requested more videos than available records')
    basename=Path(entry['name']).name
    names=sorted(records,key=lambda n:(hashlib.sha256(('%d:%s:%s'%(seed,basename,n)).encode()).hexdigest(),n))[:videos]
    output.mkdir(parents=True)
    manifest=dict(schema_version=1,dataset='robotap',files=[],selected_videos=[],seed=seed,
        source_member_provenance=provenance,source_video_count=len(records),fps_description=fps_description,
        selection='lowest sha256(seed:shard:name) within split4 only; full frames and queryable tracks',
        scope='fixed small evaluation subset; split3 omitted for bandwidth; no outcome filtering',
        role='evaluation only; never calibration training supervision',whole_archive_md5_verified=False)
    atomic_json(output/'status.json',dict(state='preparing'))
    try:
        for number,name in enumerate(names):
            path=output/('video_%06d.pkl'%number)
            with path.open('xb') as stream:pickle.dump({name:records[name]},stream,protocol=4)
            manifest['files'].append(dict(path=path.name,sha256=file_sha256(path),split='eval',fps=fps))
            manifest['selected_videos'].append(dict(name=name,source_shard=basename))
        del records
        atomic_json(output/'manifest.pending.json',manifest)
        summaries=[dict(name=s.name,frames=len(s.rgb),queries=len(s.query_txy)) for s in iter_robotap(output/'manifest.pending.json','eval')]
        (output/'manifest.pending.json').replace(output/'manifest.json')
        atomic_json(output/'status.json',dict(state='complete',videos=summaries))
    except BaseException as error:
        atomic_json(output/'status.json',dict(state='failed',error=type(error).__name__+': '+str(error)));raise
    return output/'manifest.json'


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--member',default='data/downloads/robotap_split4.pkl')
    p.add_argument('--index',default='configs/robotap.zip-index.json');p.add_argument('--output',required=True)
    p.add_argument('--fps',type=float,required=True);p.add_argument('--fps-description',required=True)
    p.add_argument('--videos',type=int,default=3);p.add_argument('--seed',type=int,default=20261029)
    a=p.parse_args();print(prepare(a.member,a.index,a.output,a.fps,a.fps_description,a.videos,a.seed))
