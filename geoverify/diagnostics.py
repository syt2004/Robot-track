"""Explain saved causal predictions using labels only after inference.

Figures are diagnostics, not proof of universal correctness. All visible,
post-query point-frames contribute to numerical summaries. Case panels include
both the largest improvement and largest regression, explicitly selected by GT.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from .datasets import file_sha256, iter_evaluation
from .geometry_worker import atomic_json


def audit_runtime_predictions(query_txy, timestamps, arrays):
    """Reject incomplete coverage, future evidence, and invalid fusion weights."""
    query_txy=np.asarray(query_txy)
    timestamps=np.asarray(timestamps)
    t, n = len(timestamps),len(query_txy)
    active = np.arange(t)[:, None] >= query_txy[None, :, 0]
    if not np.array_equal(arrays['query_txy'], query_txy) or not np.array_equal(arrays['timestamps'], timestamps):
        raise ValueError('Diagnostic query/time identity mismatch')
    if not np.array_equal(arrays['processed'], active):
        raise ValueError('Incomplete query coverage')
    xy = np.asarray(arrays['xy'])
    candidates = np.asarray(arrays['candidates'])
    valid = np.asarray(arrays['candidate_valid'], bool)
    weights = np.asarray(arrays['fusion_weights'])
    recorded = np.asarray(arrays['fusion_weights_recorded'], bool)
    if xy.shape != (t,n,2) or candidates.shape != (t,n,4,2) or valid.shape != (t,n,4) or weights.shape != valid.shape:
        raise ValueError('Diagnostic shape mismatch')
    if recorded.shape != active.shape or not recorded[active].all():
        raise ValueError('Missing actual fusion weight evidence')
    if (not np.isfinite(xy[active]).all() or not np.isfinite(candidates[valid]).all()
            or not valid[:,:,0][active].all() or not np.isfinite(weights[active]).all()
            or (weights[active]<0).any() or not np.allclose(weights[active].sum(-1),1.,atol=1e-5)
            or (weights[active][~valid[active]]!=0).any()):
        raise ValueError('Invalid positions, candidates or weights')
    reconstructed = (np.where(valid[...,None],candidates,0)*weights[...,None]).sum(-2)
    if not np.allclose(xy[active],reconstructed[active],atol=1e-3,rtol=1e-5):
        raise ValueError('Saved fusion differs from weighted candidates')
    available = np.asarray(arrays['available_timestamp'])
    evidence = np.asarray(arrays['geometry_evidence_timestamp'])
    completed = np.asarray(arrays['geometry_completed_timestamp'])
    target = np.asarray(arrays['geometry_target_frame'])
    anchor = np.asarray(arrays['geometry_anchor_frame'])
    age = np.asarray(arrays['geometry_age'])
    if any(a.shape!=active.shape for a in (available,evidence,completed,target,anchor,age)):
        raise ValueError('Temporal diagnostic shape mismatch')
    capture = np.broadcast_to(timestamps[:,None],active.shape)
    if (not np.isfinite(available[active]).all() or not np.isfinite(evidence[active]).all()
            or (available[active]<capture[active]).any() or (evidence[active]<capture[active]).any()
            or (evidence[active]>available[active]).any()):
        raise ValueError('Invalid output/evidence availability')
    known = active & (target>=0)
    if ((target[known]!=np.floor(target[known])).any() or (anchor[known]!=np.floor(anchor[known])).any()
            or (anchor[known]<0).any() or (anchor[known]>target[known]).any()
            or (target>np.arange(t)[:,None])[known].any()):
        raise ValueError('Future or invalid geometry frame')
    if known.any():
        target_time = timestamps[target[known].astype(int)]
        if (not np.isfinite(completed[known]).all() or (completed[known]<target_time).any()
                or (completed[known]>evidence[known]+1e-6).any()
                or not np.allclose(age[known],evidence[known]-target_time,atol=1e-6)):
            raise ValueError('Geometry completion/age violates causality')
    geo_valid = valid[:,:,1:3].any(-1)
    if (geo_valid & (~known | np.asarray(arrays['geometry_expired'],bool))).any():
        raise ValueError('Missing or expired geometry is marked valid')
    return dict(active=active,geometry_valid=geo_valid,summary=dict(
        processed_point_frames=int(active.sum()),
        geometry_valid_point_frames=int((active&geo_valid).sum()),
        mean_geometry_weight=float(weights[:,:,1:3].sum(-1)[active].mean()),
        checks=['complete_query_coverage','finite_valid_candidates','normalized_masked_weights',
                'fusion_reconstruction','no_future_geometry','completion_before_evidence_use','geometry_age'],
        limitation='Saved-output invariants only; no accuracy or task-success claim.'))


def audit_predictions(sample, arrays):
    runtime=audit_runtime_predictions(sample.query_txy,sample.timestamps,arrays)
    active=runtime['active'];geo_valid=runtime['geometry_valid']
    t,n=active.shape
    candidates=np.asarray(arrays['candidates']);xy=np.asarray(arrays['xy'])
    weights=np.asarray(arrays['fusion_weights'])
    evaluated = active & (np.arange(t)[:,None]>sample.query_txy[None,:,0]) & sample.target_visible
    raw_error = np.linalg.norm(candidates[:,:,0]-sample.target_xy,axis=-1)
    fused_error = np.linalg.norm(xy-sample.target_xy,axis=-1)
    if not np.isfinite(raw_error[evaluated]).all() or not np.isfinite(fused_error[evaluated]).all():
        raise ValueError('Nonfinite labeled evaluation')
    gain = raw_error-fused_error
    def mean(a, mask):
        return float(a[mask].mean()) if mask.any() else None
    return dict(active=active,evaluated=evaluated,raw_error=raw_error,fused_error=fused_error,gain=gain,
                summary=dict(processed_point_frames=int(active.sum()),evaluated_visible_point_frames=int(evaluated.sum()),
                             raw_mean_error_pixels=mean(raw_error,evaluated),fused_mean_error_pixels=mean(fused_error,evaluated),
                             improved_point_frames=int((evaluated&(gain>1e-3)).sum()),
                             worsened_point_frames=int((evaluated&(gain< -1e-3)).sum()),
                             geometry_valid_point_frames=int((active&geo_valid).sum()),
                             mean_geometry_weight=mean(weights[:,:,1:3].sum(-1),active),
                             checks=['complete_query_coverage','finite_valid_candidates','normalized_masked_weights',
                                     'fusion_reconstruction','no_future_geometry','completion_before_evidence_use','geometry_age'],
                             limitation='Checks cover saved outputs, not every internal model operation or generalization.'))


def render_diagnostics(sample, arrays, output, method_label='GeoVerify'):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    evidence = audit_predictions(sample,arrays)
    root = Path(output)
    root.mkdir(parents=True,exist_ok=False)
    active,mask = evidence['active'],evidence['evaluated']
    times = sample.timestamps
    colors = ['#2670b7','#e28b26','#8f58af','#777777']
    labels = ['C1: Track-On','C2: delayed 2D geometry','C3: delayed 3D projection','C4: constant velocity']
    def per_frame(values,valid):
        return np.divide(np.where(valid,values,0).sum(1),valid.sum(1),
                         out=np.full(len(times),np.nan),where=valid.sum(1)>0)
    fig,axes=plt.subplots(3,2,figsize=(14,11),constrained_layout=True)
    ax=axes[0,0]
    ax.plot(times,per_frame(evidence['raw_error'],mask),label='Track-On',color=colors[0])
    ax.plot(times,per_frame(evidence['fused_error'],mask),label='Fused',color='#159b80')
    ax.set(title='Visible GT error (query frame excluded)',ylabel='Original-image pixels'); ax.legend()
    ax=axes[0,1]
    for m in range(4):
        ax.plot(times,per_frame(arrays['fusion_weights'][:,:,m],active),label=labels[m],color=colors[m])
    ax.set(title='Actual fusion weights',ylabel='Mean weight',ylim=(-.02,1.02)); ax.legend(fontsize=8)
    ax=axes[1,0]
    for m in (1,2):
        ax.plot(times,per_frame(arrays['candidate_valid'][:,:,m],active),label=labels[m],color=colors[m])
    ax.set(title='Available geometry candidates',ylabel='Fraction of active points',ylim=(-.02,1.02)); ax.legend(fontsize=8)
    ax=axes[1,1]
    known=active&(arrays['geometry_target_frame']>=0)
    ax.plot(times,per_frame(arrays['geometry_age'],known),label='Geometry age at use')
    ax.plot(times,per_frame(arrays['available_timestamp']-times[:,None],active),label='Capture-to-output latency')
    ax.set(title='Delay and stale evidence',ylabel='Seconds'); ax.legend(fontsize=8)
    ax=axes[2,0]
    gains=evidence['gain'][mask]
    if len(gains): ax.hist(gains,bins=40,color='#159b80')
    ax.axvline(0,color='black',lw=1)
    ax.set(title='All evaluated visible points; includes regressions',xlabel='Raw error minus fused error (pixels)',ylabel='Point-frames')
    ax=axes[2,1]
    confidence=np.asarray(arrays['confidence'])
    if confidence.shape!=mask.shape or not np.isfinite(confidence[active]).all() or ((confidence[active]<0)|(confidence[active]>1)).any():
        raise ValueError('Invalid confidence evidence')
    bins=[]
    for lo in np.arange(0,1,.1):
        selected=mask&(confidence>=lo)&(confidence<(lo+.1) if lo<.89 else confidence<=1.)
        if selected.any(): bins.append((float(confidence[selected].mean()),float((evidence['fused_error'][selected]<=4).mean()),int(selected.sum())))
    if bins:
        b=np.array(bins); ax.scatter(b[:,0],b[:,1],s=20+100*b[:,2]/b[:,2].max(),color='#159b80')
    ax.plot([0,1],[0,1],'--',color='gray')
    ax.set(title='Reliability diagnostic: visible GT only, <=4 native px',xlabel='Mean predicted confidence',ylabel='Empirical fused correctness',xlim=(0,1),ylim=(0,1))
    for ax in axes[:2].flat: ax.set_xlabel('Replay capture time (s)')
    fig.suptitle(sample.name+' | '+method_label+'\nCausal replay diagnostics (not single-stream throughput)',fontsize=13)
    fig.savefig(root/'diagnostics.png',dpi=160); plt.close(fig)
    # Deterministic temporal coverage and both extremes, never best-only panels.
    cases=[('Uniform frame',int(f),None) for f in np.unique(np.linspace(0,len(times)-1,4,dtype=int))]
    if mask.any():
        best_label='Largest improvement' if evidence['gain'][mask].max()>0 else 'Least regression (no improvements)'
        worst_label='Largest regression' if evidence['gain'][mask].min()<0 else 'Smallest improvement (no regressions)'
        for label,score in [(best_label,evidence['gain']),(worst_label,-evidence['gain'])]:
            f,q=np.unravel_index(np.argmax(np.where(mask,score,-np.inf)),mask.shape)
            cases.append((label,int(f),int(q)))
    fig,axes=plt.subplots(2,3,figsize=(15,9),constrained_layout=True)
    case_records=[]
    for ax,(label,f,q) in zip(axes.flat,cases):
        ax.imshow(sample.rgb[f]); ax.axis('off')
        ids=np.flatnonzero(mask[f])
        if q is not None: ids=np.array([q])
        elif len(ids)>32: ids=ids[np.linspace(0,len(ids)-1,32,dtype=int)]
        if len(ids):
            gt=sample.target_xy[f,ids]; raw=arrays['candidates'][f,ids,0]; fused=arrays['xy'][f,ids]
            ax.scatter(gt[:,0],gt[:,1],s=35,facecolors='none',edgecolors='lime',label='GT')
            ax.scatter(raw[:,0],raw[:,1],s=20,marker='x',c=colors[0],label='Track-On')
            ax.scatter(fused[:,0],fused[:,1],s=12,c='#ff42cc',label='Fused')
            for g,p in zip(gt,fused): ax.plot([g[0],p[0]],[g[1],p[1]],color='#ff42cc',lw=.7,alpha=.7)
        subtitle='' if q is None else '\nquery %d: raw %.2f / fused %.2f px'%(q,evidence['raw_error'][f,q],evidence['fused_error'][f,q])
        ax.set_title('%s | frame %d%s'%(label,f,subtitle),fontsize=10)
        ax.set_xlim(0,sample.rgb.shape[2]); ax.set_ylim(sample.rgb.shape[1],0)
        case_records.append(dict(selection=label,frame=f,query=q,displayed_query_ids=ids.tolist()))
    for ax in list(axes.flat)[len(cases):]: ax.axis('off')
    handles,legend=axes.flat[-1].get_legend_handles_labels()
    if handles: fig.legend(handles,legend,loc='outside lower center',ncol=3)
    fig.suptitle(sample.name+' | '+method_label+'\nTemporal coverage + explicitly GT-selected best/worst cases',fontsize=12)
    fig.savefig(root/'cases.png',dpi=160); plt.close(fig)
    report=dict(schema_version=1,video=sample.name,method_label=method_label,summary=evidence['summary'],cases=case_records,
                confidence_bins=[dict(mean_confidence=x,correct_fraction=y,count=n) for x,y,n in bins],
                note='GT is used only for post-inference diagnostics. Display point cap does not change numerical evaluation.',
                files={name:file_sha256(root/name) for name in ('diagnostics.png','cases.png')})
    atomic_json(root/'diagnostics.json',report)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-manifest',required=True)
    parser.add_argument('--prediction-manifest',required=True)
    parser.add_argument('--split',required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    from .evaluate import evaluate
    evaluate(args.dataset_manifest,args.prediction_manifest,args.split)
    manifest_path=Path(args.prediction_manifest).resolve()
    manifest=json.loads(manifest_path.read_text())
    entries={v['name']:v for v in manifest['videos']}
    root=Path(args.output); root.mkdir(parents=True,exist_ok=False)
    for i,sample in enumerate(iter_evaluation(args.dataset_manifest,args.split)):
        with np.load(manifest_path.parent/entries[sample.name]['path'],allow_pickle=False) as a:
            render_diagnostics(sample,a,root/('video_%06d'%i))


if __name__=='__main__':
    main()
