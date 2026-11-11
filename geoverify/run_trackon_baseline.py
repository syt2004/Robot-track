"""Run the real frozen temporary Track-On2 baseline on complete selected videos.

This is independent baseline evidence, never full GeoVerify acceptance. Labels
are read by the dataset adapter, but only RGB and initial queries enter Track-On.
"""
import argparse
import json
from pathlib import Path
import time
import numpy as np
from .datasets import iter_evaluation,file_sha256
from .metrics import tapvid_first
from .evaluate import aggregate_video_metrics
from .geometry_worker import atomic_json
from .overlay import render_overlay


def run(dataset_manifest,split,output,device='cuda:0',root='.'):
    root=Path(root).resolve(); output=Path(output).resolve()
    output.mkdir(parents=True,exist_ok=False)
    report=dict(state='running',stage='asset_check',full_framework=False,model='trackon2_dinov2',
                split=split,project_training_scene_diagnostic=split=='train',
                geometry_loaded=False,calibrator_loaded=False,dataset_manifest_sha256=file_sha256(dataset_manifest),videos=[])
    atomic_json(output/'status.json',report)
    try:
        from scripts.check_assets import check_asset
        lock=json.loads((root/'configs/assets.lock.json').read_text())
        for key in ('trackon2_dinov2','dinov2_small','dinov2_config'):
            if check_asset(root,lock['downloads'][key])['status']!='verified':
                raise ValueError('Unverified baseline asset: '+key)
        from .adapters import TrackOnAdapter
        model=TrackOnAdapter(root/'third_party/track_on',root/lock['downloads']['trackon2_dinov2']['path'],
                             device,'trackon2_dinov2',root/'weights/dinov2-small')
        report.update(stage='inference',source_commit=lock['sources']['track_on']['commit'],
                      weights_sha256=lock['downloads']['trackon2_dinov2']['sha256'],
                      backbone_sha256=lock['downloads']['dinov2_small']['sha256'])
        all_metrics={}
        for index,sample in enumerate(iter_evaluation(dataset_manifest,split)):
            t,n=sample.target_visible.shape
            xy=np.broadcast_to(sample.query_txy[:,1:],(t,n,2)).copy()
            probability=np.zeros((t,n)); processed=np.zeros((t,n),bool)
            duration=np.full((t,n),np.nan)
            started=time.monotonic()
            for start in np.unique(sample.query_txy[:,0]).astype(int):
                ids=np.flatnonzero(sample.query_txy[:,0]==start)
                model.reset()
                for frame in range(start,t):
                    tick=time.monotonic()
                    pred=model.step(sample.rgb[frame],sample.query_txy[ids,1:] if frame==start else None)
                    xy[frame,ids]=pred['xy']; probability[frame,ids]=pred['visibility']
                    duration[frame,ids]=time.monotonic()-tick; processed[frame,ids]=True
                    if (frame+1)%20==0: print(sample.name,'cohort',start,'frame',frame+1,'/',t,flush=True)
            expected=np.arange(t)[:,None]>=sample.query_txy[None,:,0]
            if not np.array_equal(processed,expected) or not np.isfinite(xy).all():
                raise ValueError('Incomplete or invalid baseline predictions')
            if not np.isfinite(probability[processed]).all() or ((probability[processed]<0)|(probability[processed]>1)).any():
                raise ValueError('Invalid baseline visibility')
            visible=(probability>=.5)&processed
            scores=tapvid_first(sample.query_txy,sample.target_xy,sample.target_visible,xy,visible,
                                sample.rgb.shape[2],sample.rgb.shape[1])
            all_metrics[sample.name]=scores
            folder=output/('video_%06d'%index); folder.mkdir()
            np.savez_compressed(folder/'tracks.npz',xy=xy,visibility=probability,visible=visible,
                                query_txy=sample.query_txy,timestamps=sample.timestamps,processed=processed,
                                processing_seconds=duration)
            render_overlay(sample.rgb,sample.timestamps,xy,visible,probability,folder/'overlay.mp4',
                           'Track-On2 baseline ONLY | visibility probability')
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            fig,axes=plt.subplots(2,3,figsize=(14,9),constrained_layout=True)
            for ax,frame in zip(axes.flat,np.linspace(0,t-1,6,dtype=int)):
                ax.imshow(sample.rgb[frame]); ax.axis('off'); ax.set_title('Frame %d'%frame)
                ids=np.flatnonzero(sample.target_visible[frame]&processed[frame])
                if len(ids)>32:ids=ids[np.linspace(0,len(ids)-1,32,dtype=int)]
                if len(ids):
                    gt=sample.target_xy[frame,ids]; p=xy[frame,ids]
                    ax.scatter(gt[:,0],gt[:,1],facecolors='none',edgecolors='lime',s=30)
                    ax.scatter(p[:,0],p[:,1],c='#ff42cc',marker='x',s=20)
                    for a,b in zip(gt,p):ax.plot([a[0],b[0]],[a[1],b[1]],c='#ff42cc',lw=.7)
                ax.set_xlim(0,sample.rgb.shape[2]);ax.set_ylim(sample.rgb.shape[1],0)
            fig.suptitle(sample.name+' / '+split+' | Track-On2 baseline ONLY | green: GT, magenta: prediction\nUniform frames; at most 32 display points; metrics use all selected queries')
            fig.savefig(folder/'cases.png',dpi=160);plt.close(fig)
            entry=dict(name=sample.name,frames=t,queries=n,seconds=time.monotonic()-started,
                       inference_seconds_sum_unique_calls=float(sum(duration[f,np.flatnonzero(sample.query_txy[:,0]==s)[0]]
                       for s in np.unique(sample.query_txy[:,0]).astype(int) for f in range(s,t))),
                       metrics={k:float(v) if np.isfinite(v) else None for k,v in scores.items()},
                       files={p.name:file_sha256(p) for p in folder.iterdir()})
            report['videos'].append(entry); atomic_json(output/'status.json',report)
        means,counts=aggregate_video_metrics(all_metrics)
        report.update(state='complete',stage='finished',metrics=means,defined_video_counts=counts,
                      note='Full selected-video baseline only; no geometry/calibrator. Cohort replay is not single-stream throughput.')
        atomic_json(output/'status.json',report)
    except BaseException as error:
        report.update(state='failed',error=type(error).__name__+': '+str(error))
        atomic_json(output/'status.json',report); raise
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset-manifest',required=True);p.add_argument('--split',required=True)
    p.add_argument('--output',required=True);p.add_argument('--device',default='cuda:0');p.add_argument('--root',default='.')
    a=p.parse_args(); run(a.dataset_manifest,a.split,a.output,a.device,a.root)
