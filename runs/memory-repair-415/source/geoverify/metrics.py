"""TAP-Vid first-query metric equations, independently implemented in numpy.

Protocol reference: google-deepmind/tapnet/tapnet/tapvid/evaluation_datasets.py.
Per-video metrics are averaged equally across videos, not weighted by length.
"""
import numpy as np


def tapvid_first(query_txy,target_xy,target_visible,pred_xy,pred_visible,width,height):
    gt=np.asarray(target_xy,float)
    pred=np.asarray(pred_xy,float)
    visible=np.asarray(target_visible,bool)
    predicted=np.asarray(pred_visible,bool)
    query=np.asarray(query_txy,float)
    if gt.ndim!=3 or gt.shape[-1]!=2 or pred.shape!=gt.shape:
        raise ValueError('Tracks must be [T,N,2]')
    t,n,_=gt.shape
    if visible.shape!=(t,n) or predicted.shape!=(t,n) or query.shape!=(n,3):
        raise ValueError('Metric input shape mismatch')
    if width<=0 or height<=0 or not np.isfinite(query).all():
        raise ValueError('Invalid image size or queries')
    start=query[:,0]
    if ((start<0)|(start>=t)|(start!=np.floor(start))).any():
        raise ValueError('Invalid query frame')
    evaluate=np.arange(t)[:,None]>start[None,:]
    if not np.isfinite(gt[visible&evaluate]).all() or not np.isfinite(pred[evaluate]).all():
        raise ValueError('Finite evaluated tracks required (including predicted occluded points)')
    distance=np.sum(((pred-gt)*np.array([256./width,256./height]))**2,axis=-1)
    def ratio(numerator,denominator):
        return float(numerator/denominator) if denominator else float('nan')
    result={'occlusion_accuracy':ratio(((visible==predicted)&evaluate).sum(),evaluate.sum())}
    positive=(visible&evaluate).sum()
    for threshold in (1,2,4,8,16):
        within=distance<threshold**2
        correct=within&visible
        result['pts_within_'+str(threshold)]=ratio((correct&evaluate).sum(),positive)
        tp=(correct&predicted&evaluate).sum()
        fp=((~visible|~within)&predicted&evaluate).sum()
        result['jaccard_'+str(threshold)]=ratio(tp,positive+fp)
    result['average_jaccard']=float(np.mean([result['jaccard_'+str(k)] for k in (1,2,4,8,16)]))
    result['average_pts_within_thresh']=float(np.mean([result['pts_within_'+str(k)] for k in (1,2,4,8,16)]))
    return result
