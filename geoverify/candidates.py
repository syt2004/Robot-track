"""Four-candidate construction shared by cache generation and streaming.

Geometry endpoints are converted to corrections at their original target frame,
then propagated against the current Track-On output. No history is rewritten.
"""
from collections import OrderedDict, deque
import math
import numpy as np
from geoverify.temporal import Correction, CorrectionBuffer

CANDIDATE_NAMES = ('trackon','geometry_2d','geometry_projected','constant_velocity')


class CandidateBuilder:
    def __init__(self,query_ids,history_size=256,tau_seconds=.25,max_age_seconds=1.):
        ids = np.asarray(query_ids)
        if ids.ndim!=1 or not len(ids) or ids.dtype.kind not in 'iu' or len(np.unique(ids))!=len(ids):
            raise ValueError('Unique integer query IDs required')
        if history_size<2:
            raise ValueError('At least two history frames required')
        self.query_ids=ids.copy()
        self.indices={int(q):i for i,q in enumerate(ids)}
        self.history_size=history_size
        self.history=OrderedDict()
        self.fused=deque(maxlen=2)
        self.buffers=[CorrectionBuffer(tau_seconds,max_age_seconds) for _ in range(2)]
        self.geometry_visibility=np.zeros(len(ids))
        self.geometry_target_time=None
        self.last_frame=-1
        self.last_time=-math.inf
        self.awaiting_commit=False
        self.rigid_corrected=np.zeros(len(ids),bool)
        self.rigid_window=None

    def observe_trackon(self,frame_id,timestamp,xy,visibility):
        """Record raw output before accepting any completed geometry job."""
        xy,visibility=np.asarray(xy,float),np.asarray(visibility,float)
        if self.awaiting_commit:
            raise RuntimeError('Commit the previous fused frame before advancing')
        if frame_id<=self.last_frame or not math.isfinite(timestamp) or timestamp<=self.last_time:
            raise ValueError('Frames and timestamps must increase')
        if xy.shape!=(len(self.query_ids),2) or not np.isfinite(xy).all():
            raise ValueError('Track-On must provide finite [N,2] positions')
        if visibility.shape!=(len(xy),) or not np.isfinite(visibility).all() or ((visibility<0)|(visibility>1)).any():
            raise ValueError('Track-On visibility must be [N] probabilities')
        self.history[frame_id]=(timestamp,xy.copy(),visibility.copy())
        while len(self.history)>self.history_size:
            self.history.popitem(last=False)
        self.last_frame,self.last_time=frame_id,timestamp
        self.awaiting_commit=True

    def accept_geometry(self,result,now):
        if not self.awaiting_commit:
            raise RuntimeError('Observe the current frame before accepting geometry')
        if not math.isfinite(now) or now<self.last_time:
            raise ValueError('Invalid result availability time')
        if result.target_frame>self.last_frame or result.completed_timestamp>now:
            raise ValueError('Cannot accept future geometry')
        if result.target_frame not in self.history:
            return False  # Too late for a correctly matched raw target frame.
        target_time,raw,_=self.history[result.target_frame]
        if abs(target_time-result.target_timestamp)>1e-6:
            raise ValueError('Geometry capture timestamp differs from source frame')
        if self.buffers[0].latest is not None and result.target_frame<=self.buffers[0].latest.target_frame:
            return False
        ids=np.asarray(result.query_ids)
        if ids.ndim!=1 or len(np.unique(ids))!=len(ids) or any(int(q) not in self.indices for q in ids):
            raise ValueError('Unknown or duplicate geometry query identity')
        indices=np.array([self.indices[int(q)] for q in ids],dtype=int)
        a=result.arrays
        probability=np.asarray(a['visconf'][-1],float)
        if probability.shape!=(len(ids),) or not np.isfinite(probability).all() or ((probability<0)|(probability>1)).any():
            raise ValueError('Invalid geometry visibility')
        corrections=[]
        for key,mask_key in (('xy_2d','valid_2d'),('xy_projected','valid_3d')):
            endpoint=np.asarray(a[key][-1],float)
            mask=np.asarray(a[mask_key][-1])
            if endpoint.shape!=(len(ids),2) or mask.shape!=(len(ids),) or not np.isin(mask,[0,1]).all():
                raise ValueError('Invalid sparse geometry endpoint shape or mask')
            mask=mask.astype(bool)
            if not np.isfinite(endpoint[mask]).all():
                raise ValueError('Non-finite valid geometry endpoint')
            delta=np.zeros_like(raw)
            valid=np.zeros(len(raw),bool)
            delta[indices[mask]]=endpoint[mask]-raw[indices[mask]]
            valid[indices[mask]]=True
            corrections.append(Correction(result.anchor_frame,result.target_frame,target_time,
                               result.completed_timestamp,delta,valid,result.job_id))
        # Validate both candidates before mutating either cache.
        if not 0<=result.anchor_frame<=result.target_frame or not target_time<=result.completed_timestamp:
            raise ValueError('Invalid geometry time ordering')
        for buffer,correction in zip(self.buffers,corrections):
            buffer.accept(correction,self.last_frame,now)
        self.geometry_visibility.fill(0)
        self.geometry_visibility[indices]=probability
        self.geometry_target_time=target_time
        self.rigid_corrected.fill(False)
        self.rigid_window=None
        return True

    def refine_rigid_projection(self,result,motions,now):
        """Refine only C3 for the accepted window, preserving its age/visibility.

        Rigid predictions for occluded or geometrically inconsistent endpoints
        enter the same delayed correction mechanism, not a fifth candidate.
        """
        latest=self.buffers[1].latest
        if not self.awaiting_commit or latest is None:
            raise RuntimeError('Accept a geometry window before rigid refinement')
        if (latest.window_id!=result.job_id or latest.target_frame!=result.target_frame
                or latest.completed_timestamp!=result.completed_timestamp):
            raise ValueError('Rigid refinement must match the accepted geometry window')
        if self.rigid_window==result.job_id:
            return False
        delta=latest.delta_xy.copy()
        valid=latest.valid.copy()
        changed=np.zeros(len(self.query_ids),bool)
        raw=self.history[latest.target_frame][1]
        for motion in motions:
            evidence=motion.get('rigid_target_evidence')
            if evidence is None:
                continue
            if (motion['window_id']!=result.job_id or evidence['target_frame']!=result.target_frame
                    or evidence['available_timestamp']!=result.completed_timestamp):
                raise ValueError('Mismatched rigid target evidence')
            ids=motion['query_ids']
            occluded=np.asarray(evidence['occluded_prediction_mask'])
            rejected=np.asarray(evidence['visible_rejected_mask'])
            available=np.asarray(evidence['prediction_valid'])
            if any(m.shape!=(len(ids),) or not np.isin(m,[0,1]).all() for m in (occluded,rejected,available)):
                raise ValueError('Invalid rigid evidence masks')
            if len(evidence['projected_xy'])!=len(ids):
                raise ValueError('Invalid rigid projected point count')
            for local in np.flatnonzero(available.astype(bool)&(occluded.astype(bool)|rejected.astype(bool))):
                if ids[local] not in self.indices:
                    raise ValueError('Unknown rigid query identity')
                index=self.indices[ids[local]]
                if changed[index]:
                    raise ValueError('Duplicate rigid query correction')
                point=np.asarray(evidence['projected_xy'][local],float)
                if point.shape!=(2,) or not np.isfinite(point).all():
                    raise ValueError('Invalid rigid projection')
                delta[index]=point-raw[index]
                valid[index]=True
                changed[index]=True
        replacement=CorrectionBuffer(self.buffers[1].tau,self.buffers[1].max_age)
        replacement.accept(Correction(latest.anchor_frame,latest.target_frame,latest.target_timestamp,
                           latest.completed_timestamp,delta,valid,latest.window_id),self.last_frame,now)
        self.buffers[1]=replacement
        self.rigid_corrected=changed
        self.rigid_window=result.job_id
        return bool(changed.any())

    def build(self,now):
        if not self.awaiting_commit or not math.isfinite(now) or now<self.last_time:
            raise ValueError('Observe a frame and provide a valid availability time')
        timestamp,raw,visibility=self.history[self.last_frame]
        n=len(raw)
        candidates=np.full((n,4,2),np.nan)
        valid=np.zeros((n,4),bool)
        probabilities=np.zeros((n,4))
        candidates[:,0],valid[:,0],probabilities[:,0]=raw,True,visibility
        for j,buffer in enumerate(self.buffers,1):
            xy,mask=buffer.apply(raw,now)
            candidates[mask,j]=xy[mask]
            valid[:,j]=mask
            probabilities[mask,j]=self.geometry_visibility[mask]
        if len(self.fused)==2:
            t0,p0,v0=self.fused[0]
            t1,p1,v1=self.fused[1]
            candidates[:,3]=p1+(p1-p0)*(timestamp-t1)/(t1-t0)
            # Hidden fused positions are extrapolations, not observations.
            # Feeding them back creates an unconstrained velocity recurrence.
            probabilities[:,3]=np.minimum(v0,v1)
            valid[:,3]=np.isfinite(candidates[:,3]).all(1)&(probabilities[:,3]>=.5)
            candidates[~valid[:,3],3]=np.nan
            probabilities[~valid[:,3],3]=0.
        age=0. if self.geometry_target_time is None else now-self.geometry_target_time
        latest=self.buffers[0].latest
        return dict(candidates=candidates,candidate_visibility=probabilities,valid=valid,
                    rigid_correction_mask=self.rigid_corrected&valid[:,2],
                    geometry_age=age,geometry_valid=valid[:,1:3].copy(),
                    geometry_anchor_frame=-1 if latest is None else latest.anchor_frame,
                    geometry_target_frame=-1 if latest is None else latest.target_frame,
                    geometry_completed_timestamp=np.nan if latest is None else latest.completed_timestamp,
                    geometry_expired=False if latest is None else age>self.buffers[0].max_age,
                    query_ids=self.query_ids.copy(),frame_id=self.last_frame,timestamp=timestamp)

    def commit(self,xy,visibility):
        xy,visibility=np.asarray(xy,float),np.asarray(visibility,float)
        if not self.awaiting_commit or xy.shape!=(len(self.query_ids),2) or visibility.shape!=(len(self.query_ids),):
            raise ValueError('Invalid fused result or no frame to commit')
        if not np.isfinite(xy).all() or not np.isfinite(visibility).all() or ((visibility<0)|(visibility>1)).any():
            raise ValueError('Fused result must have finite positions and visibility probabilities')
        self.fused.append((self.last_time,xy.copy(),visibility.copy()))
        self.awaiting_commit=False
