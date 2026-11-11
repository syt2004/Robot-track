"""Optional rigid aggregation in ONE Track4World window's world gauge.

Never compose these transforms across windows or send them directly to a robot.
No pose is emitted for inadequate or nonrigid support; identity is not a fallback.
"""
import math
import numpy as np
from .geometry import robust_rigid_fit,transform_points,project


def rigid_target_evidence(result,motions,image_size,reprojection_threshold=8.,visibility_threshold=.5):
    """Project fitted rigid motion at its historical target, not at arrival time.

    Occluded anchor points may have a geometric prediction without being visible.
    Visible endpoints are consistency-gated, not automatically accepted as new
    observations. This is single-window evidence, not persistent occlusion state.
    """
    if not math.isfinite(reprojection_threshold) or reprojection_threshold<=0 or not 0<=visibility_threshold<=1:
        raise ValueError('Invalid reprojection/visibility threshold')
    width,height=image_size
    if width<=0 or height<=0:
        raise ValueError('Positive image dimensions required')
    a=result.arrays
    xyz=np.asarray(a['xyz_window_world'],float)
    valid=np.asarray(a['valid_3d'],bool)
    xy=np.asarray(a['xy_2d'],float)
    valid2d=np.asarray(a['valid_2d'],bool)
    visibility=np.asarray(a['visconf'],float)
    cameras=np.asarray(a['camera_to_world'],float)
    intrinsics=np.asarray(a['intrinsics_pixels'],float)
    ids=np.asarray(result.query_ids)
    if (xyz.ndim!=3 or xyz.shape[1:]!=(len(ids),3) or len(xyz)<2
            or valid.shape!=xyz.shape[:2] or visibility.shape!=valid.shape
            or xy.shape!=valid.shape+(2,) or valid2d.shape!=valid.shape
            or cameras.shape!=(len(xyz),4,4) or intrinsics.shape!=(len(xyz),3,3)):
        raise ValueError('Invalid rigid projection geometry')
    camera=cameras[-1]
    if (not np.isfinite(camera).all() or not np.isfinite(intrinsics[-1]).all()
            or not np.allclose(camera[3],[0,0,0,1])
            or not np.allclose(camera[:3,:3].T@camera[:3,:3],np.eye(3),atol=1e-4)
            or not np.isclose(np.linalg.det(camera[:3,:3]),1.,atol=1e-4)):
        raise ValueError('Invalid target camera pose/intrinsics')
    index={int(q):i for i,q in enumerate(ids)}
    output=[]
    for motion in motions:
        entry=dict(motion)
        if entry['status']!='observed':
            entry['rigid_target_evidence']=None
            output.append(entry)
            continue
        if entry['window_id']!=result.job_id or entry['target_frame']!=result.target_frame:
            raise ValueError('Rigid pose belongs to a different geometry window')
        group=np.array([index[q] for q in entry['query_ids']],int)
        source=xyz[0,group]
        anchor_valid=valid[0,group]&np.isfinite(source).all(1)&(visibility[0,group]>=visibility_threshold)
        predicted=transform_points(source,np.asarray(entry['delta_pose'],float))
        pixels,positive=project(predicted,intrinsics[-1],np.linalg.inv(camera))
        available=anchor_valid&positive&np.isfinite(pixels).all(1)
        available &= (pixels[:,0]>=0)&(pixels[:,0]<width)&(pixels[:,1]>=0)&(pixels[:,1]<height)
        observed=valid2d[-1,group]&np.isfinite(xy[-1,group]).all(1)&(visibility[-1,group]>=visibility_threshold)
        compared=available&observed
        error=np.full(len(group),np.nan)
        error[compared]=np.linalg.norm(pixels[compared]-xy[-1,group][compared],axis=1)
        consistent=compared&(error<=reprojection_threshold)
        # Invalid or out-of-frame endpoints are not evidence of occlusion.
        occluded=available&(visibility[-1,group]<visibility_threshold)
        entry['rigid_target_evidence']=dict(
            coordinate_frame='original_image_pixels_at_target_frame',
            target_frame=int(result.target_frame),available_timestamp=float(result.completed_timestamp),
            projected_xy=[p.tolist() if ok else None for p,ok in zip(pixels,available)],
            prediction_valid=available.tolist(),occluded_prediction_mask=occluded.tolist(),
            visible_consistent_mask=consistent.tolist(),visible_rejected_mask=(compared&~consistent).tolist(),
            reprojection_error_pixels=[float(e) if ok else None for e,ok in zip(error,compared)],
            reprojection_threshold_pixels=float(reprojection_threshold),
            visibility_not_inferred_from_projection=True)
        output.append(entry)
    return output


def estimate_window_motion(result,object_ids,reliability,threshold=.02,min_inliers=4,
                           min_inlier_fraction=.5,visibility_threshold=.5,seed=0):
    ids=np.asarray(result.query_ids)
    objects=np.asarray(object_ids)
    reliability=np.asarray(reliability,float)
    a=result.arrays
    xyz=np.asarray(a['xyz_window_world'],float)
    valid=np.asarray(a['valid_3d'])
    probability=np.asarray(a['visconf'],float)
    n=len(ids)
    if (ids.shape!=(n,) or ids.dtype.kind not in 'iu' or len(np.unique(ids))!=n
            or objects.shape!=(n,) or objects.dtype.kind not in 'iu'
            or reliability.shape!=(n,)):
        raise ValueError('Object groups and reliability must follow geometry query IDs')
    if xyz.ndim!=3 or xyz.shape[1:]!=(n,3) or len(xyz)<2 or valid.shape!=xyz.shape[:2] or probability.shape!=valid.shape:
        raise ValueError('Invalid within-window geometry shape')
    if not np.isin(valid,[0,1]).all() or not np.isfinite(reliability).all() or ((reliability<0)|(reliability>1)).any():
        raise ValueError('Invalid motion validity/reliability')
    if not np.isfinite(probability).all() or ((probability<0)|(probability>1)).any():
        raise ValueError('Invalid geometry visibility confidence')
    if (not math.isfinite(threshold) or threshold<=0 or min_inliers<3
            or not 0<min_inlier_fraction<=1 or not 0<=visibility_threshold<=1):
        raise ValueError('Invalid rigid-fit thresholds')
    if not bool(a.get('metric_scale_enabled',False)):
        raise ValueError('Metric rigid threshold requires explicitly enabled metric geometry')
    source,target=xyz[0],xyz[-1]
    paired=valid[0].astype(bool)&valid[-1].astype(bool)
    paired &= np.isfinite(source).all(1)&np.isfinite(target).all(1)
    paired &= (probability[0]>=visibility_threshold)&(probability[-1]>=visibility_threshold)&(reliability>0)
    weights=np.minimum(probability[0],probability[-1])*reliability
    output=[]
    for object_id in np.unique(objects[objects>=0]):
        group=objects==object_id
        eligible=group&paired
        entry=dict(object_id=int(object_id),window_id=str(result.job_id),
                   coordinate_frame='track4world_window_world',units='estimated_meters',
                   anchor_frame=int(result.anchor_frame),target_frame=int(result.target_frame),
                   target_timestamp=float(result.target_timestamp),
                   available_timestamp=float(result.completed_timestamp),
                   query_ids=ids[group].tolist(),status='insufficient_support',delta_pose=None,
                   inlier_mask=[False]*int(group.sum()),pose_confidence=0.,
                   confidence_definition='weighted_inlier_fraction_times_exp_negative_rmse_over_threshold',
                   eligible_points=int(eligible.sum()),inlier_points=0,rmse=None)
        if eligible.sum()>=min_inliers:
            try:
                pose,inliers,residual=robust_rigid_fit(source,target,np.where(eligible,weights,0),threshold,seed=seed)
            except ValueError:
                entry['status']='degenerate_or_no_consensus'
            else:
                fraction=float(weights[inliers].sum()/weights[eligible].sum())
                if inliers.sum()<min_inliers or fraction<min_inlier_fraction:
                    entry['status']='insufficient_consensus'
                else:
                    rmse=float(np.sqrt(np.average(residual[inliers]**2,weights=weights[inliers])))
                    entry.update(status='observed',delta_pose=pose.tolist(),inlier_mask=inliers[group].tolist(),
                                 pose_confidence=fraction*math.exp(-rmse/threshold),rmse=rmse,
                                 inlier_points=int(inliers.sum()))
        output.append(entry)
    return output
