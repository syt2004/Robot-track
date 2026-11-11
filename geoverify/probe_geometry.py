"""Measure real Track4World window latency before choosing causal age limits.

Run in the isolated Track4World environment with CUDA_VISIBLE_DEVICES set.
Uses RGB and a fixed grid only; no trajectory labels enter inference.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image

from .datasets import file_sha256
from .geometry_worker import atomic_json


def load_probe_rgb(root, manifest_path, split, windows):
    """Load checksum-verified RGB only, including prepared robot video inputs."""
    manifest_path=Path(manifest_path).resolve()
    manifest=json.loads(manifest_path.read_text())
    if not windows or any(w<2 or w>16 for w in windows):
        raise ValueError('Window lengths must be between 2 and 16')
    if manifest.get('role')=='qualitative_robot_operation_only':
        from .video_input import decode_video
        entry=manifest['videos'][0]
        path=Path(root)/entry['video']
        if file_sha256(path)!=entry['video_sha256']:
            raise ValueError('RGB video checksum mismatch')
        rgb,_=decode_video(path)
        if len(rgb)!=entry['frames'] or len(rgb)<max(windows):
            raise ValueError('Robot video frame count mismatch or too short')
        return rgb[:max(windows)],entry['archive_member']
    if manifest.get('dataset')!='cotracker3_kubric':
        raise ValueError('Probe expects prepared Kubric RGB or robot RGB-only manifest')
    scene=next(s for s in manifest['scenes'] if s['split']==split)
    if len(scene['frames'])<max(windows):
        raise ValueError('Scene too short')
    frames=[]
    for entry in scene['frames'][:max(windows)]:
        path=manifest_path.parent/entry['path']
        if file_sha256(path)!=entry['sha256']:
            raise ValueError('RGB checksum mismatch')
        with Image.open(path) as image:
            frames.append(np.asarray(image.convert('RGB')))
    return np.stack(frames),scene['name']


def run(root, dataset_manifest, split, output, windows,metric_only=False):
    root=Path(root).resolve()
    manifest_path=Path(dataset_manifest).resolve()
    rgb,scene_name=load_probe_rgb(root,manifest_path,split,windows)
    lock=json.loads((root/'configs/assets.lock.json').read_text())
    paths={}
    for name in (('track4world_da3','da3_config') if metric_only else ('track4world_da3','da3_nested','da3_config')):
        entry=lock['downloads'][name]
        path=root/entry['path']
        if not path.is_file() or path.stat().st_size!=entry['bytes']:
            raise ValueError('Missing or unverified geometry asset: '+name)
        if 'sha256' in entry:
            verified=file_sha256(path)==entry['sha256']
        else:
            content=path.read_bytes()
            verified=hashlib.sha1(b'blob '+str(len(content)).encode()+b'\0'+content).hexdigest()==entry['git_blob_sha1']
        if not verified:
            raise ValueError('Geometry asset checksum mismatch: '+name)
        paths[name]=path
    if metric_only:
        from .metric_bundle import audit_bundle
        selection=json.loads((root/'configs/da3.metric.lock.json').read_text())
        if selection['source_sha256']!=lock['downloads']['da3_nested']['sha256']:
            raise ValueError('Metric source does not match asset lock')
        directory=root/'weights/da3nested-metric';audit_bundle(directory,selection)
        if file_sha256(directory/'config.json')!=file_sha256(paths['da3_config']):
            raise ValueError('Metric backbone config mismatch')
        paths['da3_nested']=directory/'model.safetensors'
    h,w=rgb.shape[1:3]
    x,y=np.meshgrid(np.linspace(w*.2,w*.8,4),np.linspace(h*.2,h*.8,4))
    queries=np.stack([x.ravel(),y.ravel()],axis=-1)
    out=Path(output).resolve();out.mkdir(parents=True,exist_ok=False)
    report=dict(state='loading',scene=scene_name,dataset_manifest_sha256=file_sha256(manifest_path),
                labels_used=False,metric_only_initialization=metric_only,query_protocol='fixed 4 by 4 grid',windows=[],
                scope='Geometry frontend latency and finite-output probe; not full fusion or tracking accuracy')
    atomic_json(out/'status.json',report)
    try:
        from .track4world_adapter import Track4WorldAdapter
        import torch
        start=time.monotonic()
        model=Track4WorldAdapter(root/'third_party/track4world',paths['track4world_da3'],
                                backbone_directory=paths['da3_nested'].parent)
        torch.cuda.synchronize()
        report.update(state='running',load_seconds=time.monotonic()-start)
        atomic_json(out/'status.json',report)
        for index,length in enumerate(windows):
            torch.cuda.reset_peak_memory_stats()
            start=time.monotonic()
            arrays=model.run_window(rgb[:length],queries)
            torch.cuda.synchronize()
            seconds=time.monotonic()-start
            for xy,valid in (('xy_2d','valid_2d'),('xy_projected','valid_3d')):
                if not np.isfinite(arrays[xy][arrays[valid]]).all():
                    raise ValueError('Non-finite valid geometry output: '+xy)
            path=out/('window_%02d_%02d.npz'%(index,length))
            np.savez_compressed(path,query_xy=queries,**arrays)
            result=dict(length=length,seconds=seconds,first_forward=index==0,
                        endpoint_valid_2d=float(arrays['valid_2d'][-1].mean()),
                        endpoint_valid_3d=float(arrays['valid_3d'][-1].mean()),
                        peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                        path=path.name,sha256=file_sha256(path))
            report['windows'].append(result)
            atomic_json(out/'status.json',report)
            print(json.dumps(result),flush=True)
        report['state']='complete';atomic_json(out/'status.json',report)
    except BaseException as error:
        report.update(state='failed',error=type(error).__name__+': '+str(error))
        atomic_json(out/'status.json',report)
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',default='.')
    parser.add_argument('--dataset-manifest',required=True)
    parser.add_argument('--split',default='val')
    parser.add_argument('--output',required=True)
    parser.add_argument('--windows',type=int,nargs='+',default=[2,2,4,8,16])
    parser.add_argument('--metric-only',action='store_true')
    args=parser.parse_args()
    run(args.root,args.dataset_manifest,args.split,args.output,args.windows,args.metric_only)
