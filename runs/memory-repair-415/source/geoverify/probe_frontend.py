"""Real-weight frontend integration diagnostic, not full-framework acceptance.

Uses two RGB frames by default, or the complete scene, and a fixed query grid;
does not read trajectory labels or train on validation data. No geometry or
calibrator is substituted. Run in the isolated Track-On environment.
"""
import argparse
import hashlib
import io
import json
from pathlib import Path
import tarfile
import time
import numpy as np
from PIL import Image
from .datasets import file_sha256
from .geometry_worker import atomic_json
from .prepare_kubric import selected_members


def probe(root,output,device='cuda:0',frame_count=2):
    if frame_count!=0 and frame_count<2:
        raise ValueError('Use at least two frames, or zero for the complete scene')
    root=Path(root).resolve(strict=True)
    output=Path(output).absolute()
    output.mkdir(parents=True,exist_ok=False)
    report=dict(state='running',stage='assets',full_framework=False,
                variant='trackon2_dinov2',geometry_loaded=False,calibrator_loaded=False)
    atomic_json(output/'status.json',report)
    try:
        assets=json.loads((root/'configs/assets.lock.json').read_text())['downloads']
        paths={}
        hashes={}
        for key in ('trackon2_dinov2','verifier','dinov2_small','dinov2_config'):
            entry=assets[key]
            path=root/entry['path']
            digest=file_sha256(path)
            valid=path.stat().st_size==entry['bytes']
            if 'sha256' in entry:
                valid=valid and digest==entry['sha256']
            else:
                content=path.read_bytes()
                valid=valid and hashlib.sha1(b'blob '+str(len(content)).encode()+b'\0'+content).hexdigest()==entry['git_blob_sha1']
            if not valid:
                raise ValueError('Unverified asset: '+key)
            paths[key]=path
            hashes[key]=digest
        selection=json.loads((root/'configs/kubric.small.lock.json').read_text())
        entry=next(e for e in selection['files'] if e['split']=='val')
        archive=root/'data/downloads/kubric'/entry['path']
        if archive.stat().st_size!=entry['bytes'] or file_sha256(archive)!=entry['sha256']:
            raise ValueError('Unverified validation archive')
        scene=entry['path'][:-len('.tar.gz')]
        with tarfile.open(archive,'r:gz') as source:
            frames=sorted((m for m in selected_members(source,scene) if '/frames/' in m.name),key=lambda m:m.name)
            total_frames=len(frames)
            if frame_count:
                frames=frames[:frame_count]
            if len(frames)<2 or (frame_count and len(frames)!=frame_count):
                raise ValueError('Need two source RGB frames')
        # Gzip seeking to filename-sorted members can repeatedly decompress the
        # archive if physical member order differs. Read one sequential pass.
        wanted={m.name for m in frames}
        decoded={}
        with tarfile.open(archive,'r|gz') as source:
            for member in source:
                if member.name in wanted:
                    with source.extractfile(member) as stream:
                        with Image.open(io.BytesIO(stream.read())) as im:
                            decoded[member.name]=np.asarray(im.convert('RGB'))
        rgb=np.stack([decoded[m.name] for m in frames])
        report.update(stage='model_loading',archive_sha256=entry['sha256'],
                      frame_members=[m.name for m in frames],queries='fixed 4 by 4 image grid; no labels',
                      asset_sha256=hashes)
        atomic_json(output/'status.json',report)
        from .adapters import TrackOnAdapter,VerifierAdapter
        import torch
        tracker=TrackOnAdapter(root/'third_party/track_on',paths['trackon2_dinov2'],device,
                               'trackon2_dinov2',paths['dinov2_small'].parent)
        verifier=VerifierAdapter(root/'third_party/track_on',paths['verifier'],device)
        h,w=rgb.shape[1:3]
        x,y=np.meshgrid(np.linspace(.2*(w-1),.8*(w-1),4),np.linspace(.2*(h-1),.8*(h-1),4))
        queries=np.c_[x.ravel(),y.ravel()]
        report['stage']='forward'
        atomic_json(output/'status.json',report)
        start=time.monotonic()
        predictions=[]
        for i,frame in enumerate(rgb):
            predictions.append(tracker.step(frame,queries if i==0 else None))
            if (i+1)%10==0:
                print('Frontend diagnostic: %d/%d frames'%(i+1,len(rgb)),flush=True)
        xy=np.stack([p['xy'] for p in predictions])
        vis=np.stack([p['visibility'] for p in predictions])
        # Exercise the upstream full temporal window after a complete sequential
        # tracker rollout. This is a frontend diagnostic, not fused inference.
        window_start=max(0,len(rgb)-24)
        candidates=np.repeat(xy[window_start:,:,None,:],4,axis=2)
        valid=np.broadcast_to([True,False,False,True],candidates.shape[:-1]).copy()
        ranks=verifier.rank_window(rgb[window_start:],np.c_[np.zeros(len(queries)),xy[window_start]],candidates,valid)
        if not np.isfinite(xy).all() or not np.isfinite(vis).all() or not np.isfinite(ranks[:,[0,3]]).all() or not np.isneginf(ranks[:,[1,2]]).all():
            raise ValueError('Nonfinite frontend output or invalid candidates not masked')
        if device.startswith('cuda'):
            torch.cuda.synchronize()
        np.savez_compressed(output/'prediction.npz',xy=xy,visibility=vis,ranks=ranks,queries=queries)
        report.update(state='complete',stage='finished',frames=len(rgb),source_frames=total_frames,
                      complete_scene=len(rgb)==total_frames,verifier_window_frames=len(candidates),query_count=len(queries),
                      seconds=time.monotonic()-start,device=device,
                      prediction_sha256=file_sha256(output/'prediction.npz'))
        atomic_json(output/'status.json',report)
    except BaseException as error:
        report.update(state='failed',error=type(error).__name__+': '+str(error))
        atomic_json(output/'status.json',report)
        raise
    return output/'status.json'


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',default='.')
    parser.add_argument('--output',required=True)
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--frames',type=int,default=2,help='Zero runs every RGB frame; default two-frame startup check')
    args=parser.parse_args()
    print(probe(args.root,args.output,args.device,args.frames))
