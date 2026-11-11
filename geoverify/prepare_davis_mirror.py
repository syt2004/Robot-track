"""Prepare the fixed three-video supplementary mirror subset, without training use."""
import argparse
import hashlib
import json
from pathlib import Path
import pickle
import cv2
import numpy as np
import pyarrow.parquet as pq
from .datasets import file_sha256,iter_robotap
from .geometry_worker import atomic_json


def prepare(root,output):
    root=Path(root).resolve();output=Path(output).resolve()
    lock_path=root/'configs/tapvid-davis-mirror.lock.json'
    lock=json.loads(lock_path.read_text());source=root/lock['path']
    if source.stat().st_size!=lock['bytes'] or file_sha256(source)!=lock['sha256']:
        raise ValueError('Mirror source hash/size mismatch')
    records=pq.read_table(source).to_pylist()
    if len(records)!=lock['declared_rows']:raise ValueError('Mirror row count changed')
    seed=20261030
    selected=sorted(range(len(records)),key=lambda i:hashlib.sha256(('%d:%d'%(seed,i)).encode()).digest())[:3]
    output.mkdir(parents=True,exist_ok=False)
    atomic_json(output/'status.json',dict(state='preparing'))
    manifest=dict(schema_version=1,dataset='tapvid_davis',files=[],selected_rows=selected,seed=seed,
        source_lock_sha256=file_sha256(lock_path),source_parquet_sha256=lock['sha256'],
        source_repository=lock['repository'],source_revision=lock['revision'],
        role='supplementary evaluation only; never training',
        scope='Three fixed records from a 29-record MP4 mirror; not official full DAVIS benchmark',
        upstream_official_byte_identity_verified=False,
        fps_description='25 Hz from mirror MP4 container; original acquisition timing not independently verified',
        selection='lowest sha256(seed:row_index); full decoded frames and every supplied track')
    try:
        expected={}
        for row in selected:
            record=records[row];video_path=output/('row_%02d.mp4'%row)
            video_path.write_bytes(record['video'])
            cap=cv2.VideoCapture(str(video_path));fps=cap.get(cv2.CAP_PROP_FPS);frames=[]
            try:
                while True:
                    ok,frame=cap.read()
                    if not ok:break
                    frames.append(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
            finally:cap.release()
            if not frames or fps!=25.:raise ValueError('Unexpected mirror decoding/FPS')
            rgb=np.stack(frames);tracks=np.asarray(record['tracks'],dtype=np.float64)
            queries=np.asarray(record['query_points'],dtype=np.float64)
            visible=np.asarray(record['visibles'])
            if rgb.shape[1:]!=(256,256,3) or tracks.shape!=(len(queries),len(rgb),2) or visible.shape!=tracks.shape[:2]:
                raise ValueError('Mirror frame/track mismatch')
            if queries.shape!=(len(tracks),3) or visible.dtype!=bool or not np.isfinite(tracks).all():
                raise ValueError('Invalid mirror annotations')
            first=visible.argmax(axis=1)
            if not visible.any(axis=1).all() or not np.array_equal(queries,np.c_[first,tracks[np.arange(len(tracks)),first]]):
                raise ValueError('Mirror queries are not exact first-visible txy pixels')
            name='mirror_row_%02d'%row
            converted=dict(video=rgb,points=tracks/np.array([256.,256.]),occluded=~visible)
            path=output/(name+'.pkl')
            with path.open('xb') as f:pickle.dump({name:converted},f,protocol=4)
            manifest['files'].append(dict(path=path.name,sha256=file_sha256(path),split='eval',fps=fps,
                source_row=row,video_sha256=file_sha256(video_path),frames=len(rgb),queries=len(queries)))
            expected[name]=(queries,tracks,visible)
        atomic_json(output/'manifest.pending.json',manifest)
        for sample in iter_robotap(output/'manifest.pending.json','eval'):
            query,track,visible=expected.pop(sample.name)
            np.testing.assert_array_equal(sample.query_txy,query)
            np.testing.assert_array_equal(sample.target_xy,track.transpose(1,0,2))
            np.testing.assert_array_equal(sample.target_visible,visible.T)
        if expected:raise ValueError('Prepared records missing')
        (output/'manifest.pending.json').replace(output/'manifest.json')
        atomic_json(output/'status.json',dict(state='complete',selected_rows=selected,full_benchmark=False))
    except BaseException as error:
        atomic_json(output/'status.json',dict(state='failed',error=type(error).__name__+': '+str(error)));raise
    return output/'manifest.json'


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',default='.')
    parser.add_argument('--output',required=True)
    args=parser.parse_args();print(prepare(args.root,args.output))
