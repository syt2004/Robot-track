"""Independent causal raw-2D visibility; delayed geometry never enters this API."""
from collections import deque
from pathlib import Path
import math
import numpy as np
import torch
from .features import calibration_features
from .raw_visibility_head import GeometryCalibrationHead, HEAD_ARCHITECTURE, FEATURE_NAMES
from .datasets import file_sha256

POLICY = 'independent_raw_history_v1'

def raw_visibility_inputs(raw_xy, raw_visibility, timestamp, width, height, history):
    xy=np.asarray(raw_xy,dtype=float);visibility=np.asarray(raw_visibility,dtype=float);now=float(timestamp)
    if xy.ndim!=2 or xy.shape[1]!=2 or visibility.shape!=(len(xy),) or not len(xy):
        raise ValueError('Raw visibility requires nonempty [N,2] coordinates and [N] probabilities')
    if not np.isfinite(xy).all() or not np.isfinite(visibility).all() or ((visibility<0)|(visibility>1)).any():
        raise ValueError('Invalid raw visibility inputs')
    if not math.isfinite(now) or width<=0 or height<=0:
        raise ValueError('Invalid capture time or image size')
    history=list(history)
    if len(history)>3 or any(not math.isfinite(t) or t>=now or p.shape!=xy.shape or v.shape!=visibility.shape for t,p,v in history):
        raise ValueError('Raw history must contain only matching past observations')
    if any(history[j][0]>=history[j+1][0] for j in range(len(history)-1)):
        raise ValueError('Raw history times must increase')
    n=len(xy);candidates=np.full((n,4,2),np.nan);valid=np.zeros((n,4),bool);vis=np.zeros((n,4));logits=np.zeros((n,4),np.float32)
    candidates[:,0]=xy;valid[:,0]=True;vis[:,0]=visibility
    if len(history)>=2:
        t0,p0,v0=history[-2];t1,p1,v1=history[-1]
        candidates[:,3]=p1+(p1-p0)*(now-t1)/(t1-t0);vis[:,3]=np.minimum(v0,v1);valid[:,3]=vis[:,3]>=.5
        candidates[~valid[:,3],3]=np.nan;vis[~valid[:,3],3]=0
    packet=dict(candidates=candidates,valid=valid,candidate_visibility=vis,timestamp=now,geometry_age=0.)
    features=calibration_features(packet,logits,width,height,history=history)
    return dict(features=features,visual_logits=logits,candidates=candidates,candidate_visibility=vis,valid=valid)

class RawVisibilityStream:
    """CPU visibility model with bounded raw-only history, reset per query cohort."""
    def __init__(self,checkpoint,expected_sha256,baseline_provenance):
        checkpoint=Path(checkpoint).resolve()
        if file_sha256(checkpoint)!=expected_sha256:
            raise ValueError('Independent visibility checkpoint hash mismatch')
        c=torch.load(checkpoint,map_location='cpu',weights_only=True)
        if c.get('head_architecture')!=HEAD_ARCHITECTURE or c.get('feature_names')!=list(FEATURE_NAMES):
            raise ValueError('Independent visibility architecture/schema mismatch')
        if c.get('schema_version')!=1 or not isinstance(c.get('epoch'),int) or c['epoch']<1 or not c.get('manifest_sha256'):
            raise ValueError('Independent visibility requires a trained bound checkpoint')
        if c.get('baseline_provenance')!=baseline_provenance:
            raise ValueError('Independent visibility baseline provenance mismatch')
        self.model=GeometryCalibrationHead().cpu().eval();self.model.load_state_dict(c['model'],strict=True);self.model.requires_grad_(False)
        if not all(bool(torch.isfinite(v).all()) for v in self.model.state_dict().values()) or not bool((self.model.feature_std>0).all()):
            raise ValueError('Invalid independent visibility checkpoint values')
        self.checkpoint_sha256=expected_sha256;self.training_source_sha256=c['training_source_sha256'];self.reset()
    def reset(self):
        self.history=deque(maxlen=3);self.image_size=None;self.last_inputs=None
    def predict(self,raw_xy,raw_visibility,timestamp,width,height):
        size=(int(width),int(height))
        if self.image_size is not None and self.image_size!=size:
            raise ValueError('Raw visibility image size changed inside a cohort')
        arrays=raw_visibility_inputs(raw_xy,raw_visibility,timestamp,width,height,self.history)
        keys=['features','visual_logits','candidates','candidate_visibility','valid']
        tensors=[torch.as_tensor(arrays[k]).bool() if k=='valid' else torch.as_tensor(arrays[k]).float() for k in keys]
        with torch.inference_mode():prob=self.model(*tensors)['visibility'].numpy().copy()
        if not np.isfinite(prob).all() or ((prob<0)|(prob>1)).any():
            raise ValueError('Non-finite independent visibility prediction')
        self.history.append((float(timestamp),np.asarray(raw_xy,dtype=float).copy(),np.asarray(raw_visibility,dtype=float).copy()))
        self.image_size=size;self.last_inputs=arrays
        return prob

def replay_raw_visibility(stream,raw_xy,raw_visibility,timestamps,query_txy,width,height,max_queries_per_cohort=None):
    """Replay saved raw inputs only; never read fused trajectories or labels."""
    xy=np.asarray(raw_xy);vis=np.asarray(raw_visibility);times=np.asarray(timestamps);queries=np.asarray(query_txy)
    if xy.ndim!=3 or xy.shape[-1]!=2 or vis.shape!=xy.shape[:2] or times.shape!=(len(xy),) or queries.shape!=(xy.shape[1],3):
        raise ValueError('Invalid raw replay dimensions')
    starts=queries[:,0]
    if not np.isfinite(starts).all() or ((starts<0)|(starts>=len(xy))|(starts!=np.floor(starts))).any():
        raise ValueError('Invalid raw query start frames')
    result=np.zeros(vis.shape,dtype=np.float32)
    for start in np.unique(starts).astype(int):
        ids=np.flatnonzero(starts==start);size=len(ids) if max_queries_per_cohort is None else int(max_queries_per_cohort)
        if size<1:raise ValueError('Invalid raw replay query cohort limit')
        for offset in range(0,len(ids),size):
            group=ids[offset:offset+size];stream.reset()
            for frame in range(start,len(xy)):
                result[frame,group]=stream.predict(xy[frame,group],vis[frame,group],times[frame],width,height)
    return result
