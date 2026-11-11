"""Shared causal RGB pipeline; production callers supply official adapters.

All query identities are initialized at the first supplied frame. Late-query
benchmark cohorts require separate instances until dynamic insertion is added.
No model fallback or ground-truth input exists in this interface.
"""
from collections import deque
import math
import numpy as np
from geoverify.candidates import CandidateBuilder
from geoverify.features import calibration_features,build_neighbor_graph


class CausalGeoVerifyTracker:
    def __init__(self,trackon,verifier,calibrator,geometry_worker,
                 geometry_window=16,geometry_stride=4,verifier_window=24,
                 tau_seconds=.25,max_geometry_age=1.,clock=None,rigid_threshold=None,
                 rigid_reprojection_threshold=8.,evidence_age_clock='availability',
                 evidence_history_policy='fused'):
        if not 2<=geometry_window<=16 or geometry_stride<1 or verifier_window<1:
            raise ValueError('Invalid history window or stride')
        if any(x is None for x in (trackon,verifier,calibrator,geometry_worker)):
            raise ValueError('The full architecture requires every model component')
        self.trackon,self.verifier,self.calibrator,self.worker=trackon,verifier,calibrator,geometry_worker
        self.window,self.stride,self.verifier_window=geometry_window,geometry_stride,verifier_window
        self.tau,self.max_age=tau_seconds,max_geometry_age
        self.evidence_age_clock=evidence_age_clock
        if evidence_history_policy not in ('fused','raw_trackon'):
            raise ValueError('Unknown evidence history policy')
        self.evidence_history_policy=evidence_history_policy
        # Live callers provide a monotonic clock mapped into the capture-time
        # domain. Explicit availability times support deterministic replay.
        self.clock=clock
        self.frames=deque(maxlen=max(geometry_window,verifier_window))
        self.packets=deque(maxlen=verifier_window)
        self.fused=deque(maxlen=max(geometry_window,verifier_window))
        self.raw_history=deque(maxlen=max(geometry_window,verifier_window))
        self.builder=None
        self.next_frame=0
        self.last_geometry_submit=-geometry_stride
        self.last_available=-math.inf
        self.depth_change=None
        self.depth_valid=None
        if rigid_threshold is not None and (not math.isfinite(rigid_threshold) or rigid_threshold<=0):
            raise ValueError('Rigid threshold must be positive in estimated meters')
        self.rigid_threshold=rigid_threshold
        if not math.isfinite(rigid_reprojection_threshold) or rigid_reprojection_threshold<=0:
            raise ValueError('Rigid reprojection threshold must be positive pixels')
        self.rigid_reprojection_threshold=rigid_reprojection_threshold

    def initialize(self,rgb,queries,timestamp=0.,object_ids=None,available_timestamp=None):
        if self.builder is not None:
            raise RuntimeError('Create a new tracker for a new video')
        queries=np.asarray(queries,float)
        if queries.ndim!=2 or queries.shape[1]!=2 or not len(queries) or not np.isfinite(queries).all():
            raise ValueError('Finite nonempty [N,2] query coordinates required')
        self.query_ids=np.arange(len(queries),dtype=np.int64)
        self.object_ids=np.full(len(queries),-1) if object_ids is None else np.asarray(object_ids).copy()
        if self.object_ids.shape!=(len(queries),) or self.object_ids.dtype.kind not in 'iu':
            raise ValueError('One integer object ID per query required')
        self.neighbors=build_neighbor_graph(queries,self.object_ids)
        self.builder=CandidateBuilder(self.query_ids,tau_seconds=self.tau,max_age_seconds=self.max_age,evidence_age_clock=self.evidence_age_clock)
        self.trackon.reset()
        if getattr(self.calibrator,'visibility_stream',None) is not None:
            self.calibrator.visibility_stream.reset()
        return self._step(rgb,timestamp,available_timestamp,new_queries=queries)

    def step(self,rgb,timestamp,available_timestamp=None):
        if self.builder is None:
            raise RuntimeError('Initialize queries first')
        return self._step(rgb,timestamp,available_timestamp)

    def _depth_evidence(self,result):
        xyz=np.asarray(result.arrays['xyz_camera'])
        valid=np.asarray(result.arrays['valid_3d'],bool)
        n=len(self.query_ids)
        changes=np.zeros(n)
        available=np.zeros(n,bool)
        if len(xyz)<2:
            return changes,available
        ids=np.asarray(result.query_ids,dtype=int)
        previous=np.full(n,np.nan)
        current=np.full(n,np.nan)
        previous[ids]=xyz[-2,:,2]
        current[ids]=xyz[-1,:,2]
        available[ids]=valid[-2]&valid[-1]
        available &= np.isfinite(previous)&np.isfinite(current)&(previous>0)&(current>0)
        for i in np.flatnonzero(available):
            group=np.r_[i,self.neighbors[i]]
            group=group[available[group]]
            denominator=np.median(previous[group])
            changes[i]=abs(current[i]-previous[i])/max(denominator,1e-8)
        return changes,available

    def _step(self,rgb,timestamp,available_timestamp,new_queries=None):
        rgb=np.asarray(rgb)
        now=float(timestamp if available_timestamp is None else available_timestamp)
        if not math.isfinite(timestamp) or (self.frames and timestamp<=self.frames[-1][1]):
            raise ValueError('Capture timestamps must be finite and strictly increasing')
        if self.clock is not None:
            now=max(now,float(self.clock()))
        if rgb.ndim!=3 or rgb.shape[-1]!=3 or rgb.dtype!=np.uint8:
            raise ValueError('Expected RGB uint8 HWC frame')
        if not math.isfinite(now) or now<timestamp or now<self.last_available:
            raise ValueError('Availability clock must be monotonic and not before capture')
        if self.frames and rgb.shape!=self.frames[-1][2].shape:
            raise ValueError('Image size cannot change mid-video')
        frame_id=self.next_frame
        raw=self.trackon.step(rgb,new_queries=new_queries)
        if self.clock is not None:
            observed=float(self.clock())
            if not math.isfinite(observed) or observed<now:
                raise ValueError('Live clock moved backwards')
            now=observed
        self.builder.observe_trackon(frame_id,timestamp,raw['xy'],raw['visibility'])
        self.frames.append((frame_id,timestamp,rgb.copy()))
        evidence_history=self.raw_history if self.evidence_history_policy=='raw_trackon' else self.fused
        result=self.worker.poll(frame_id,now)
        accepted=result is not None and self.builder.accept_geometry(result,now)
        if accepted:
            self.depth_change,self.depth_valid=self._depth_evidence(result)
        packet=self.builder.build(now)
        motions=[]
        if accepted and not packet['geometry_expired'] and self.rigid_threshold is not None and (self.object_ids>=0).any():
            from geoverify.object_motion import estimate_window_motion,rigid_target_evidence
            # Use only previously published endpoint reliability, never GT or
            # the not-yet-computed current fusion output. Missing history => 0.
            history_by_frame={item['frame_id']:item for item in evidence_history}
            reliability=np.zeros(len(self.query_ids))
            if result.anchor_frame in history_by_frame and result.target_frame in history_by_frame:
                reliability=np.minimum(history_by_frame[result.anchor_frame]['confidence'],
                                       history_by_frame[result.target_frame]['confidence'])
            ids=np.asarray(result.query_ids,dtype=int)
            motions=estimate_window_motion(result,self.object_ids[ids],reliability[ids],self.rigid_threshold)
            motions=rigid_target_evidence(result,motions,(rgb.shape[1],rgb.shape[0]),
                                          self.rigid_reprojection_threshold)
            self.builder.refine_rigid_projection(result,motions,now)
            packet=self.builder.build(now)
        self.packets.append(packet)
        evidence_timestamp=now
        length=len(self.packets)
        video=np.stack([v[2] for v in list(self.frames)[-length:]])
        candidates=np.stack([v['candidates'] for v in self.packets])
        masks=np.stack([v['valid'] for v in self.packets])
        # Use the previously emitted anchor if this is a rolling window; no GT.
        anchor_id=list(self.frames)[-length][0]
        if anchor_id==frame_id:
            anchor=np.asarray(raw['xy'])
        else:
            anchor=next(x['xy'] for x in evidence_history if x['frame_id']==anchor_id)
        queries=np.c_[np.zeros(len(anchor)),anchor]
        scores=self.verifier.rank_window(video,queries,candidates,valid=masks)
        if getattr(self.calibrator,'requires_descriptor',False):
            descriptor=np.asarray(self.verifier.last_descriptor)
            if descriptor.shape!=(len(packet['valid']),4,256) or not np.isfinite(descriptor).all() or np.any(descriptor[~packet['valid']]!=0):
                raise ValueError('Current-frame visual descriptor missing or misaligned')
            packet['appearance_descriptor']=descriptor.copy()
        history=[(x['timestamp'],x['xy'],x['confidence']) for x in list(evidence_history)[-3:]]
        features=calibration_features(packet,scores,rgb.shape[1],rgb.shape[0],history,
                                      self.neighbors,self.depth_change,self.depth_valid)
        prediction=self.calibrator.predict(features,scores,packet)
        if getattr(self.calibrator,'visibility_stream',None) is not None:
            prediction['visibility']=self.calibrator.visibility_stream.predict(
                raw['xy'],raw['visibility'],timestamp,rgb.shape[1],rgb.shape[0])
        # Raw-conditioned evidence does not recycle a learned correction into
        # future candidate/history/anchor inputs. Published xy is still calibrated.
        if self.evidence_history_policy=='raw_trackon':
            self.builder.commit(raw['xy'],raw['visibility'])
        else:
            self.builder.commit(prediction['xy'],prediction['visibility'])
        if self.clock is not None:
            observed=float(self.clock())
            if not math.isfinite(observed) or observed<now:
                raise ValueError('Live clock moved backwards')
            now=observed
        output=dict(frame_id=frame_id,timestamp=float(timestamp),available_timestamp=now,
                    fusion_completed_timestamp=now,geometry_evidence_timestamp=evidence_timestamp,
                    xy=prediction['xy'].copy(),visibility=prediction['visibility'].copy(),
                    confidence=prediction['confidence'].copy(),source=getattr(self.calibrator,'output_kind','geoverify'),
                    tracker_source=raw.get('source','unspecified'),
                    geometry_used=bool(packet['geometry_valid'].any()),
                    geometry_used_mask=packet['geometry_valid'].any(axis=1),
                    candidates=packet['candidates'],valid=packet['valid'],
                    candidate_visibility=packet['candidate_visibility'],
                    rigid_correction_mask=packet['rigid_correction_mask'],
                    features=features,visual_logits=scores,object_motion=motions)
        for key in ('geometry_age','geometry_anchor_frame','geometry_target_frame',
                    'geometry_completed_timestamp','geometry_expired','geometry_wall_age','geometry_capture_age','geometry_age_clock'):
            output[key]=packet[key]
        if 'appearance_descriptor' in packet:
            output['appearance_descriptor']=packet['appearance_descriptor'].copy()
        if 'weights' in prediction:
            output['fusion_weights']=np.asarray(prediction['weights']).copy()
        self.fused.append(dict(frame_id=frame_id,timestamp=float(timestamp),
                               xy=output['xy'].copy(),confidence=output['confidence'].copy()))
        self.raw_history.append(dict(frame_id=frame_id,timestamp=float(timestamp),
                                     xy=np.asarray(raw['xy'],dtype=float).copy(),
                                     confidence=np.asarray(raw['visibility'],dtype=float).copy()))
        self.next_frame+=1
        if len(self.frames)>=self.window and frame_id-self.last_geometry_submit>=self.stride:
            window=list(self.frames)[-self.window:]
            start=window[0][0]
            anchor=next(x['xy'] for x in evidence_history if x['frame_id']==start)
            submitted=self.worker.submit(np.stack([x[2] for x in window]),anchor,self.query_ids,
                                         [x[0] for x in window],[x[1] for x in window],now)
            if submitted:
                self.last_geometry_submit=frame_id
        # Submission serializes/memmaps the window before returning. Include
        # that time in actual publication latency, not only GPU fusion time.
        if self.clock is not None:
            observed=float(self.clock())
            if not math.isfinite(observed) or observed<now:
                raise ValueError('Live clock moved backwards during geometry submission')
            now=observed
        output['available_timestamp']=now
        self.last_available=now
        return output

    def close(self):
        self.worker.close()
