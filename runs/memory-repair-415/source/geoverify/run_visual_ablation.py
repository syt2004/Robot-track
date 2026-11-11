"""Explicit no-geometry ablation: Track-On + frozen Verifier + constant velocity.

Not a fallback for missing geometry assets and never full-method acceptance.
Uses the same causal runner and candidate history, with geometry disabled by
design. Fusion is original Verifier softmax; confidence is untrained 0.5.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from .bootstrap import VerifierBootstrapPolicy
from .datasets import iter_evaluation,file_sha256
from .diagnostics import render_diagnostics
from .evaluate import aggregate_video_metrics
from .geometry_worker import atomic_json
from .metrics import tapvid_first
from .overlay import render_overlay
from .pipeline import CausalGeoVerifyTracker
from .video_runner import run_query_cohorts,save_video_predictions


class DisabledGeometryAblation:
    """No model, no synthetic geometry and no completed geometry result."""
    def poll(self,current_frame,now):return None
    def submit(self,*args,**kwargs):return False
    def close(self):pass


class VisualAblationPolicy(VerifierBootstrapPolicy):
    output_kind='ablation_visual_verifier_no_geometry_no_calibrator'


def run(dataset_manifest,split,output,root='.',device='cuda:0'):
    root=Path(root).resolve();output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False)
    label='ABLATION: visual Verifier + CV; NO geometry/calibrator; confidence fixed 0.5'
    report=dict(state='loading',full_framework=False,method=label,split=split,
                dataset_manifest_sha256=file_sha256(dataset_manifest),videos=[],
                geometry_loaded=False,calibrator_loaded=False)
    atomic_json(output/'status.json',report)
    try:
        from scripts.check_assets import check_asset
        lock=json.loads((root/'configs/assets.lock.json').read_text())
        hashes={}
        for name in ('trackon2_dinov2','dinov2_small','dinov2_config','verifier'):
            entry=lock['downloads'][name];result=check_asset(root,entry)
            if result['status']!='verified':raise ValueError('Unverified asset: '+name)
            hashes[name]=file_sha256(root/entry['path'])
        from .adapters import TrackOnAdapter,VerifierAdapter
        trackon=TrackOnAdapter(root/'third_party/track_on',root/lock['downloads']['trackon2_dinov2']['path'],
                               device,'trackon2_dinov2',root/'weights/dinov2-small')
        verifier=VerifierAdapter(root/'third_party/track_on',root/lock['downloads']['verifier']['path'],device)
        report.update(state='running',asset_sha256=hashes,source_commit=lock['sources']['track_on']['commit'])
        atomic_json(output/'status.json',report)
        def factory(clock):
            return CausalGeoVerifyTracker(trackon,verifier,VisualAblationPolicy(),DisabledGeometryAblation(),clock=clock)
        records={};raw_records={}
        for index,sample in enumerate(iter_evaluation(dataset_manifest,split)):
            print('Ablation',sample.name,split,len(sample.rgb),'frames',len(sample.query_txy),'queries',flush=True)
            def progress(frame,ids,packet):
                if (frame+1)%20==0:print('Frame',frame+1,'/',len(sample.rgb),flush=True)
            arrays=run_query_cohorts(sample.rgb,sample.timestamps,sample.query_txy,factory,
                                     record_candidates=True,on_output=progress)
            if arrays['candidate_valid'][:,:,1:3].any():raise ValueError('Geometry appeared in no-geometry ablation')
            folder=output/('video_%06d'%index);folder.mkdir()
            save_video_predictions(folder/'tracks.npz',arrays)
            shape=(sample.rgb.shape[2],sample.rgb.shape[1])
            records[sample.name]=tapvid_first(sample.query_txy,sample.target_xy,sample.target_visible,
                                             arrays['xy'],arrays['visible'],*shape)
            raw_records[sample.name]=tapvid_first(sample.query_txy,sample.target_xy,sample.target_visible,
                arrays['candidates'][:,:,0],arrays['candidate_visibility'][:,:,0]>=.5,*shape)
            diagnostic=render_diagnostics(sample,arrays,folder/'diagnostics',method_label=label+' / '+split)
            render_overlay(sample.rgb,sample.timestamps,arrays['xy'],arrays['visible'],arrays['confidence'],
                           folder/'overlay.mp4','VISUAL ABLATION ONLY | confidence fixed 0.5')
            report['videos'].append(dict(name=sample.name,path=str(folder.relative_to(output)),
                tracks_sha256=file_sha256(folder/'tracks.npz'),metrics=records[sample.name],
                raw_metrics=raw_records[sample.name],diagnostic_summary=diagnostic['summary']))
            atomic_json(output/'status.json',report)
        report['metrics'],report['defined_video_counts']=aggregate_video_metrics(records)
        report['raw_metrics'],_=aggregate_video_metrics(raw_records)
        report['state']='complete';atomic_json(output/'status.json',report)
    except BaseException as error:
        report.update(state='failed',error=type(error).__name__+': '+str(error))
        atomic_json(output/'status.json',report);raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset-manifest',required=True);p.add_argument('--split',required=True)
    p.add_argument('--output',required=True);p.add_argument('--root',default='.')
    p.add_argument('--device',default='cuda:0');a=p.parse_args()
    run(a.dataset_manifest,a.split,a.output,a.root,a.device)
