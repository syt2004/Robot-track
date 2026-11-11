"""Optional SE(3) gauge alignment using same-frame camera poses, never object motion.

Requires explicitly metric windows; this does not estimate Sim(3) scale or prove
metric depth calibration. Static-camera overlap cannot independently verify scale.
No persistent world history is changed by this function.
"""
import math
import numpy as np


def _cameras(ids,poses):
    ids,poses=np.asarray(ids),np.asarray(poses,float)
    if ids.ndim!=1 or ids.dtype.kind not in 'iu' or len(np.unique(ids))!=len(ids) or (ids<0).any():
        raise ValueError('Unique nonnegative integer camera frame IDs required')
    if poses.shape!=(len(ids),4,4) or not np.isfinite(poses).all():
        raise ValueError('Finite camera-to-world [T,4,4] poses required')
    rotations=poses[:,:3,:3]
    if (not np.allclose(poses[:,3],np.array([0,0,0,1]),atol=1e-6)
            or not np.allclose(rotations@rotations.transpose(0,2,1),np.eye(3),atol=1e-4)
            or not np.allclose(np.linalg.det(rotations),1.,atol=1e-4)):
        raise ValueError('Camera poses must be proper SE(3) transforms')
    return ids,poses


def align_overlapping_cameras(reference_ids,reference_poses,window_ids,window_poses,*,
                              metric_scale_enabled,translation_threshold=.03,
                              rotation_threshold_degrees=3.,min_overlap=3,min_inlier_fraction=.6):
    reference_ids,reference_poses=_cameras(reference_ids,reference_poses)
    window_ids,window_poses=_cameras(window_ids,window_poses)
    if (not math.isfinite(translation_threshold) or translation_threshold<=0
            or not math.isfinite(rotation_threshold_degrees) or not 0<rotation_threshold_degrees<180
            or not isinstance(min_overlap,int) or min_overlap<2 or not 0<min_inlier_fraction<=1):
        raise ValueError('Invalid camera alignment thresholds')
    common,ri,wi=np.intersect1d(reference_ids,window_ids,return_indices=True)
    result=dict(status='unaligned',window_to_reference=None,overlap_frame_ids=common.tolist(),
                inlier_mask=[False]*len(common),reason=None,scale_estimated=False)
    if not metric_scale_enabled:
        result['reason']='metric_scale_not_enabled'
        return result
    if len(common)<min_overlap:
        result['reason']='insufficient_same_frame_overlap'
        return result
    reference,window=reference_poses[ri],window_poses[wi]
    # T_reference_world_from_camera @ T_camera_from_window_world.
    hypotheses=reference@np.linalg.inv(window)
    angle_limit=math.radians(rotation_threshold_degrees)
    def errors(transform):
        predicted=transform[None]@window
        translation=np.linalg.norm(predicted[:,:3,3]-reference[:,:3,3],axis=1)
        delta=predicted[:,:3,:3]@reference[:,:3,:3].transpose(0,2,1)
        angle=np.arccos(np.clip((np.trace(delta,axis1=1,axis2=2)-1)/2,-1,1))
        return translation,angle
    best=None
    for transform in hypotheses:
        translation,angle=errors(transform)
        mask=(translation<=translation_threshold)&(angle<=angle_limit)
        score=(int(mask.sum()),-float((translation[mask]/translation_threshold+angle[mask]/angle_limit).sum()))
        if best is None or score>best[0]:
            best=(score,mask)
    mask=best[1]
    if mask.sum()<min_overlap or mask.mean()<min_inlier_fraction:
        result['reason']='inconsistent_overlap'
        return result
    u,_,vt=np.linalg.svd(hypotheses[mask,:3,:3].sum(axis=0))
    correction=np.eye(3); correction[-1,-1]=np.linalg.det(u@vt)
    rotation=u@correction@vt
    translation=np.median(reference[mask,:3,3]-(rotation@window[mask,:3,3].T).T,axis=0)
    transform=np.eye(4); transform[:3,:3]=rotation; transform[:3,3]=translation
    residual,angle=errors(transform)
    inliers=(residual<=translation_threshold)&(angle<=angle_limit)
    if inliers.sum()<min_overlap or inliers.mean()<min_inlier_fraction:
        result['reason']='refit_failed_consensus'
        return result
    result.update(status='aligned',window_to_reference=transform.tolist(),inlier_mask=inliers.tolist(),
                  translation_rmse=float(np.sqrt(np.mean(residual[inliers]**2))),
                  rotation_rmse_degrees=float(np.degrees(np.sqrt(np.mean(angle[inliers]**2)))))
    return result
