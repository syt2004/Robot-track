"""Full-video, causal first-query replay without feeding labels to the model.

Each distinct query-start frame runs a fresh full architecture on the remaining
video. This preserves every query without changing the fixed-query fusion state.
Cohort replay is explicitly NOT a throughput measurement of a single online
session; it repeats computation. Each cohort itself is capture-time paced and
records actual output availability, including processing backlog.
"""
import time
import copy
import json
from pathlib import Path
import numpy as np


def run_query_cohorts(rgb,timestamps,query_txy,tracker_factory,
                     visibility_threshold=.5,object_ids=None,
                     monotonic=time.monotonic,sleep=time.sleep,on_output=None,
                     record_candidates=False,max_queries_per_cohort=None):
    """factory(clock=callable) returns the complete CausalGeoVerifyTracker.

    Only RGB, timestamps, initial queries and optional externally supplied
    object IDs cross this interface. GT trajectories/occlusion are not accepted.
    Model construction happens before starting each cohort's capture clock.
    """
    rgb=np.asarray(rgb)
    timestamps=np.asarray(timestamps,float)
    query=np.asarray(query_txy,float)
    if rgb.ndim!=4 or rgb.shape[-1]!=3 or rgb.dtype!=np.uint8 or len(rgb)==0:
        raise ValueError('Nonempty RGB uint8 THWC video required')
    if timestamps.shape!=(len(rgb),) or not np.isfinite(timestamps).all() or (np.diff(timestamps)<=0).any():
        raise ValueError('Strictly increasing finite video timestamps required')
    if query.ndim!=2 or query.shape[1]!=3 or not len(query) or not np.isfinite(query).all():
        raise ValueError('Nonempty finite [N,3] queries required')
    starts=query[:,0]
    if ((starts<0)|(starts>=len(rgb))|(starts!=np.floor(starts))).any():
        raise ValueError('Query frame outside video or not an integer')
    if not np.isfinite(visibility_threshold) or not 0<=visibility_threshold<=1:
        raise ValueError('Visibility threshold must be in [0,1]')
    if object_ids is not None:
        object_ids=np.asarray(object_ids)
        if object_ids.shape!=(len(query),):
            raise ValueError('Object IDs must match query order')
    if max_queries_per_cohort is not None:
        if isinstance(max_queries_per_cohort,bool) or not isinstance(max_queries_per_cohort,(int,np.integer)) or max_queries_per_cohort<1:
            raise ValueError('Query cohort limit must be a positive integer')
        if object_ids is not None:
            raise ValueError('Query batching cannot split externally grouped rigid objects')
    t,n=len(rgb),len(query)
    xy=np.broadcast_to(query[:,1:],(t,n,2)).copy()
    probability=np.zeros((t,n))
    confidence=np.zeros((t,n))
    available=np.full((t,n),np.nan)
    processing_seconds=np.full((t,n),np.nan)
    input_backlog_seconds=np.full((t,n),np.nan)
    processed=np.zeros((t,n),bool)
    geometry_used=np.zeros((t,n),bool)
    rigid_corrected=np.zeros((t,n),bool)
    diagnostics={}
    if record_candidates:
        diagnostics=dict(candidates=np.zeros((t,n,4,2)),
                         fusion_weights=np.full((t,n,4),np.nan),
                         fusion_weights_recorded=np.zeros((t,n),bool),
                         fusion_completed_timestamp=np.full((t,n),np.nan),
                         geometry_evidence_timestamp=np.full((t,n),np.nan),
                         candidate_valid=np.zeros((t,n,4),bool),
                         candidate_visibility=np.zeros((t,n,4)),
                         geometry_anchor_frame=np.full((t,n),-1,np.int64),
                         geometry_target_frame=np.full((t,n),-1,np.int64),
                         geometry_age=np.full((t,n),np.nan),
                         geometry_completed_timestamp=np.full((t,n),np.nan),
                         geometry_expired=np.zeros((t,n),bool))
    sources=set()
    motions=[]
    cohort_count=0
    def cohorts():
        for start in np.unique(starts).astype(int):
            ids=np.flatnonzero(starts==start)
            size=len(ids) if max_queries_per_cohort is None else max_queries_per_cohort
            for offset in range(0,len(ids),size):
                yield start,ids[offset:offset+size]
    for start,ids in cohorts():
        origin=[None]
        def clock():
            if origin[0] is None:
                raise RuntimeError('Capture clock used before model construction finished')
            return timestamps[start]+monotonic()-origin[0]
        tracker=tracker_factory(clock=clock)
        try:
            origin[0]=monotonic()
            for frame in range(start,t):
                # Never let accelerated file decoding outrun the asynchronous
                # geometry worker's physical completion clock.
                while True:
                    delay=timestamps[frame]-clock()
                    if delay<=0:
                        break
                    sleep(min(max(float(delay),1e-6),.05))
                input_backlog=max(0.,clock()-timestamps[frame])
                processing_start=monotonic()
                if frame==start:
                    output=tracker.initialize(rgb[frame],query[ids,1:],timestamp=timestamps[frame],
                                              object_ids=None if object_ids is None else object_ids[ids])
                else:
                    output=tracker.step(rgb[frame],timestamp=timestamps[frame])
                duration=monotonic()-processing_start
                if not np.isfinite(duration) or duration<0:
                    raise ValueError('Invalid monotonic processing duration')
                for key,shape in (('xy',(len(ids),2)),('visibility',(len(ids),)),('confidence',(len(ids),))):
                    values=np.asarray(output[key])
                    if values.shape!=shape or not np.isfinite(values).all():
                        raise ValueError('Invalid pipeline output: '+key)
                    if key!='xy' and ((values<0)|(values>1)).any():
                        raise ValueError('Pipeline probability outside [0,1]')
                observed=float(output['available_timestamp'])
                if not np.isfinite(observed) or observed<timestamps[frame] or (frame>start and observed<available[frame-1,ids[0]]):
                    raise ValueError('Invalid output availability timestamp')
                xy[frame,ids]=output['xy']
                probability[frame,ids]=output['visibility']
                confidence[frame,ids]=output['confidence']
                available[frame,ids]=observed
                processing_seconds[frame,ids]=duration
                input_backlog_seconds[frame,ids]=input_backlog
                processed[frame,ids]=True
                used=np.asarray(output.get('geometry_used_mask',
                                           np.full(len(ids),bool(output['geometry_used']))))
                if used.shape!=(len(ids),) or not np.isin(used,[0,1]).all():
                    raise ValueError('Invalid per-query geometry usage mask')
                geometry_used[frame,ids]=used.astype(bool)
                rigid=np.asarray(output.get('rigid_correction_mask',np.zeros(len(ids),bool)))
                if rigid.shape!=(len(ids),) or not np.isin(rigid,[0,1]).all() or (rigid.astype(bool)&~used.astype(bool)).any():
                    raise ValueError('Invalid rigid correction usage mask')
                rigid_corrected[frame,ids]=rigid.astype(bool)
                sources.add(output['tracker_source'])
                for motion in output.get('object_motion',[]):
                    event=copy.deepcopy(motion)
                    event['anchor_frame']=int(event['anchor_frame']+start)
                    event['target_frame']=int(event['target_frame']+start)
                    if event.get('rigid_target_evidence') is not None:
                        event['rigid_target_evidence']['target_frame']=int(event['rigid_target_evidence']['target_frame']+start)
                    event['query_ids']=ids[np.asarray(event['query_ids'],dtype=int)].tolist()
                    event.update(emitted_frame=frame,cohort_start_frame=int(start),
                                 emitted_timestamp=observed)
                    motions.append(event)
                if record_candidates:
                    for key in ('fusion_completed_timestamp','geometry_evidence_timestamp'):
                        if key in output:
                            value=float(output[key])
                            if not np.isfinite(value) or not timestamps[frame]<=value<=observed:
                                raise ValueError('Invalid pipeline phase timestamp: '+key)
                            diagnostics[key][frame,ids]=value
                    if 'fusion_weights' in output:
                        weights=np.asarray(output['fusion_weights'],float)
                        mask=np.asarray(output['valid'],bool)
                        if (weights.shape!=(len(ids),4) or mask.shape!=weights.shape
                                or not np.isfinite(weights).all() or (weights<0).any()
                                or not np.allclose(weights.sum(-1),1.,atol=1e-5)
                                or (weights[~mask]!=0).any()):
                            raise ValueError('Invalid actual fusion weights')
                        diagnostics['fusion_weights'][frame,ids]=weights
                        diagnostics['fusion_weights_recorded'][frame,ids]=True
                    target=output.get('geometry_target_frame',-1)
                    if target!=-1:
                        anchor=output['geometry_anchor_frame']
                        completed=float(output['geometry_completed_timestamp'])
                        age=float(output['geometry_age'])
                        expired=output['geometry_expired']
                        if (not isinstance(target,(int,np.integer)) or not isinstance(anchor,(int,np.integer))
                                or not 0<=anchor<=target<=frame-start
                                or not np.isfinite(completed) or not timestamps[start+target]<=completed<=observed
                                or not np.isfinite(age) or age<0 or expired not in (False,True)):
                            raise ValueError('Invalid geometry window diagnostic')
                        if expired and used.any():
                            raise ValueError('Expired geometry cannot be used')
                        diagnostics['geometry_anchor_frame'][frame,ids]=start+anchor
                        diagnostics['geometry_target_frame'][frame,ids]=start+target
                        diagnostics['geometry_age'][frame,ids]=age
                        diagnostics['geometry_completed_timestamp'][frame,ids]=completed
                        diagnostics['geometry_expired'][frame,ids]=expired
                    for source,target,shape in (
                        ('candidates','candidates',(len(ids),4,2)),
                        ('valid','candidate_valid',(len(ids),4)),
                        ('candidate_visibility','candidate_visibility',(len(ids),4))):
                        value=np.asarray(output[source])
                        if value.shape!=shape:
                            raise ValueError('Invalid candidate diagnostic: '+source)
                        if source=='candidates':
                            mask=np.asarray(output['valid'])
                            if mask.shape!=(len(ids),4) or not np.isin(mask,[0,1]).all():
                                raise ValueError('Invalid candidate mask')
                            if not np.isfinite(value[mask.astype(bool)]).all():
                                raise ValueError('Non-finite valid candidate')
                        elif not np.isfinite(value).all():
                            raise ValueError('Invalid candidate diagnostic: '+source)
                        if source=='valid' and not np.isin(value,[0,1]).all():
                            raise ValueError('Invalid candidate mask')
                        if source=='candidate_visibility' and ((value<0)|(value>1)).any():
                            raise ValueError('Invalid candidate probability')
                        diagnostics[target][frame,ids]=value
                if on_output is not None:
                    # Observer receives owned snapshots, cannot revise causal
                    # history through aliases. Its return value is ignored.
                    snapshot=copy.deepcopy(output)
                    on_output(frame,ids.copy(),snapshot)
        finally:
            tracker.close()
        cohort_count+=1
    expected=np.arange(t)[:,None]>=starts[None,:]
    if not np.array_equal(processed,expected) or len(sources)!=1:
        raise RuntimeError('Incomplete or mixed-backbone video output')
    return dict(query_txy=query.copy(),timestamps=timestamps.copy(),xy=xy,
                visible=(probability>=visibility_threshold)&processed,visibility=probability,
                confidence=confidence,available_timestamp=available,processed=processed,
                processing_seconds=processing_seconds,input_backlog_seconds=input_backlog_seconds,
                cohort_geometry_used=geometry_used,
                rigid_correction_mask=rigid_corrected,
                tracker_source=np.asarray(next(iter(sources))),
                replay_protocol=np.asarray('capture_paced_start_frame_cohorts' if max_queries_per_cohort is None else 'capture_paced_bounded_query_cohorts'),
                max_queries_per_cohort=np.asarray(-1 if max_queries_per_cohort is None else max_queries_per_cohort),
                cohort_count=np.asarray(cohort_count),
                object_motion_json=np.asarray(json.dumps(motions,allow_nan=False)),
                visibility_threshold=np.asarray(visibility_threshold),**diagnostics)


def fusion_usage(arrays):
    """Distinguish candidate availability from recorded numerical fusion weight."""
    active=np.asarray(arrays['processed'],bool)
    recorded=np.asarray(arrays['fusion_weights_recorded'],bool)
    weights=np.asarray(arrays['fusion_weights'],float)
    if recorded.shape!=active.shape or weights.shape!=active.shape+(4,):
        raise ValueError('Fusion usage diagnostic shape mismatch')
    known=active&recorded
    complete=bool(active.any() and recorded[active].all())
    if not np.isfinite(weights[known]).all() or (weights[known]<0).any() or not np.allclose(weights[known].sum(-1),1.,atol=1e-5):
        raise ValueError('Invalid recorded fusion weight values')
    geometry=weights[active,1:3].sum(-1) if complete else None
    return dict(processed_point_frames=int(active.sum()),recorded_weight_point_frames=int(known.sum()),
                weights_complete=complete,
                geometry_candidate_available_point_frames=int(np.asarray(arrays['cohort_geometry_used'],bool)[active].sum()),
                geometry_positive_weight_point_frames=int((geometry>0).sum()) if complete else None,
                mean_geometry_weight=float(geometry.mean()) if complete else None,
                interpretation='Nonzero numerical weight is not evidence of accuracy improvement')


def save_video_predictions(path,arrays):
    """Exclusive creation: a new run must never overwrite old experiment data."""
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('xb') as stream:
        np.savez_compressed(stream,**arrays)
