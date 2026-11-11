"""Causal calibration evidence, identical for cache generation and deployment."""
import math
import numpy as np
from geoverify.feature_schema import FEATURE_NAMES


def build_neighbor_graph(query_xy,object_ids,k=8):
    """Only caller-provided same-object memberships; unknown (-1) has no edges.

    Never derive object membership from ground-truth trajectories. Grouped-point
    use is not the query-independent TAP-Vid protocol and must be labeled as such.
    """
    points,objects=np.asarray(query_xy,float),np.asarray(object_ids)
    if points.ndim!=2 or points.shape[1]!=2 or not np.isfinite(points).all() or objects.shape!=(len(points),) or k<1:
        raise ValueError('Invalid neighbor graph input')
    graph=[]
    for i in range(len(points)):
        ids=np.flatnonzero((objects==objects[i]) & (objects[i]>=0) & (np.arange(len(points))!=i))
        distance=np.linalg.norm(points[ids]-points[i],axis=-1)
        graph.append(ids[np.argsort(distance,kind='stable')[:k]])
    return graph


def weighted_median(values,weights):
    order=np.argsort(values,kind='stable')
    cumulative=np.cumsum(weights[order])
    return values[order[np.searchsorted(cumulative,cumulative[-1]/2,side='left')]]


def calibration_features(packet,visual_logits,width,height,history=(),neighbors=None,
                         depth_change=None,depth_valid=None):
    """history: up to three PREVIOUS (capture_timestamp, fused_xy, reliability).

    Pixel distances normalized by image diagonal; velocities per second,
    acceleration per second squared. Missing residuals are -1 (outside their
    physical range), geometry_valid=0, and invalid candidates are masked later.
    Depth change must already be relative to a local median depth in ONE gauge.
    """
    if width<=0 or height<=0:
        raise ValueError('Image size must be positive')
    xy=np.asarray(packet['candidates'],float)
    valid=np.asarray(packet['valid'],bool)
    visibility=np.asarray(packet['candidate_visibility'],float)
    scores=np.asarray(visual_logits,float)
    if xy.ndim!=3 or xy.shape[1:]!=(4,2):
        raise ValueError('Expected four candidates [N,4,2]')
    n=len(xy)
    if valid.shape!=(n,4) or visibility.shape!=(n,4) or scores.shape!=(n,4):
        raise ValueError('Candidate shape mismatch')
    if not valid[:,0].all() or not np.isfinite(xy[valid]).all() or not np.isfinite(scores[valid]).all():
        raise ValueError('Valid candidate data must be finite and include Track-On')
    if not np.isfinite(visibility[valid]).all() or ((visibility[valid]<0)|(visibility[valid]>1)).any():
        raise ValueError('Invalid candidate visibility')
    diagonal=math.hypot(width,height)
    now=float(packet['timestamp'])
    age=float(packet['geometry_age'])
    if not math.isfinite(now) or not math.isfinite(age) or age<0:
        raise ValueError('Invalid time metadata')
    features=np.full((n,4,len(FEATURE_NAMES)),-1.,dtype=np.float32)
    features[...,0]=np.where(valid,scores,0)
    features[...,1]=visibility[:,0,None]
    features[...,2]=np.where(valid[:,1],visibility[:,1],-1)[:,None]
    for slot,reference in ((3,0),(4,1),(5,2)):
        available=valid & valid[:,reference,None]
        distance=np.linalg.norm(xy-xy[:,reference,None,:],axis=-1)/diagonal
        features[...,slot][available]=distance[available]
    geometric=valid[:,1]&valid[:,2]
    residual=np.linalg.norm(xy[:,1]-xy[:,2],axis=-1)/diagonal
    features[geometric,:,6]=residual[geometric,None]
    if (depth_change is None)!=(depth_valid is None):
        raise ValueError('Depth change and validity must be supplied together')
    if depth_change is not None:
        depth_change,depth_valid=np.asarray(depth_change,float),np.asarray(depth_valid,bool)
        if depth_change.shape!=(n,) or depth_valid.shape!=(n,):
            raise ValueError('Depth feature shape mismatch')
        if not np.isfinite(depth_change[depth_valid]).all() or (depth_change[depth_valid]<0).any():
            raise ValueError('Valid depth changes must be finite and nonnegative')
        mask=geometric&depth_valid
        features[mask,:,7]=depth_change[mask,None]
    features[...,11]=age
    features[...,12]=geometric[:,None]
    previous=list(history)[-3:]
    for t,p,c in previous:
        p,c=np.asarray(p),np.asarray(c)
        if not math.isfinite(t) or t>=now or p.shape!=(n,2) or c.shape!=(n,):
            raise ValueError('History must contain only previous frames with matching query identities')
        if not np.isfinite(p).all() or not np.isfinite(c).all() or ((c<0)|(c>1)).any():
            raise ValueError('Invalid history positions/reliability')
    if any(previous[j][0]>=previous[j+1][0] for j in range(len(previous)-1)):
        raise ValueError('History timestamps must increase')
    if len(previous)>=2:
        t0,p0,_=previous[-2]
        t1,p1,_=previous[-1]
        p0,p1=np.asarray(p0),np.asarray(p1)
        velocity=(xy-p1[:,None])/(now-t1)
        prior_velocity=(p1-p0)/(t1-t0)
        features[...,8]=np.linalg.norm(velocity-prior_velocity[:,None],axis=-1)/diagonal
        if len(previous)>=3:
            tm,pm,_=previous[-3]
            previous_velocity=(p0-np.asarray(pm))/(t0-tm)
            prior_acceleration=2*(prior_velocity-previous_velocity)/(t1-tm)
            acceleration=2*(velocity-prior_velocity[:,None])/(now-t0)
            features[...,9]=np.linalg.norm(acceleration-prior_acceleration[:,None],axis=-1)/diagonal
    if previous and neighbors is not None:
        if len(neighbors)!=n:
            raise ValueError('Neighbor graph/query count mismatch')
        _,p,confidence=previous[-1]
        displacement=xy-np.asarray(p)[:,None,:]
        confidence=np.asarray(confidence)
        # Reference motion comes from current raw Track-On neighbors only, not GT.
        for i,ids in enumerate(neighbors):
            ids=np.asarray(ids,dtype=int)
            if ((ids<0)|(ids>=n)|(ids==i)).any():
                raise ValueError('Invalid neighbor index')
            ids=ids[confidence[ids]>0]
            if not len(ids):
                continue
            reference=np.array([weighted_median(displacement[ids,0,d],confidence[ids]) for d in range(2)])
            features[i,:,10]=np.linalg.norm(displacement[i]-reference,axis=-1)/diagonal
    features[~valid]=0
    if not np.isfinite(features).all():
        raise ValueError('Non-finite calibration feature')
    return features
