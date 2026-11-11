"""Sparse extraction from Track4World's unforced pairwise output.

The upstream pair fields contain ABSOLUTE endpoints, not displacements.
Camera poses are camera-to-window-world; world coordinates from different
windows must not be directly joined. This module does not perform alignment.
"""
import numpy as np
from geoverify.geometry import bilinear_sample, project, transform_points


def extract_pairwise(geometry, motion, query_xy, *, force_projection,
                     aligned_scene_flow):
    """Inputs are CPU numpy versions of infer_pair's two dictionaries, B=1.

    Returns full within-window sparse histories; callers may only publish the
    endpoint once the window job completes. 3D candidates remain conditional
    on the 2D association, not statistically independent observations.
    """
    if force_projection or aligned_scene_flow:
        raise ValueError('Geometry verification requires unforced, unaligned raw 3D predictions')
    flow = np.asarray(motion['flow_2d'])
    endpoints = np.asarray(motion['flow_3d'])
    confidence = np.asarray(motion['visconf_maps_e'])
    points = np.asarray(geometry['points'])
    masks = np.asarray(geometry['mask'])
    intrinsics = np.asarray(geometry['intrinsics'])
    poses = np.asarray(geometry['camera_poses'])
    query = np.asarray(query_xy, dtype=float)
    if flow.ndim != 5 or flow.shape[0] != 1 or flow.shape[2] != 2:
        raise ValueError('Expected pair flow [1,T-1,2,H,W]')
    _, pairs, _, h, w = flow.shape
    t = pairs + 1
    if pairs < 1 or query.ndim != 2 or query.shape[1] != 2:
        raise ValueError('Need at least two frames and [N,2] queries')
    for value, shape in ((endpoints,(1,pairs,h,w,3)),
                         (confidence,(1,pairs,2,h,w)), (points,(1,t,h,w,3)),
                         (masks,(1,t,h,w)), (intrinsics,(1,t,3,3)), (poses,(t,4,4))):
        if value.shape != shape:
            raise ValueError('Unexpected Track4World output shape: '+str(value.shape))
    if not np.isfinite(intrinsics).all() or not np.isfinite(poses).all():
        raise ValueError('Invalid camera calibration')
    k = intrinsics[0].astype(float).copy()
    # Upstream unproject_cv uses x/W,y/H (not align_corners normalization).
    k[:,0,:] *= w
    k[:,1,:] *= h
    n = len(query)
    xy = np.full((t,n,2), np.nan)
    projected = np.full_like(xy, np.nan)
    xyz = np.full((t,n,3), np.nan)
    world = np.full_like(xyz, np.nan)
    valid2d = np.zeros((t,n),bool)
    valid3d = np.zeros_like(valid2d)
    visconf = np.zeros((t,n))
    xyz[0], initial = bilinear_sample(points[0,0], query)
    mask0, _ = bilinear_sample(masks[0,0,...,None].astype(float), query)
    valid2d[0] = initial
    xy[0,initial] = query[initial]
    projected[0], positive = project(xyz[0], k[0], np.eye(4))
    valid3d[0] = initial & positive & (mask0[:,0] >= 0.999)
    world[0] = transform_points(xyz[0], poses[0])
    visconf[0] = valid3d[0].astype(float)
    for j in range(pairs):
        next_xy, spatial = bilinear_sample(flow[0,j].transpose(1,2,0), xy[j])
        target3d, _ = bilinear_sample(endpoints[0,j], xy[j])
        probability, _ = bilinear_sample(confidence[0,j].transpose(1,2,0), xy[j])
        source_mask, _ = bilinear_sample(masks[0,j,...,None].astype(float), xy[j])
        alive = valid2d[j] & spatial & np.isfinite(next_xy).all(1)
        alive &= (next_xy[:,0]>=0) & (next_xy[:,0]<=w-1)
        alive &= (next_xy[:,1]>=0) & (next_xy[:,1]<=h-1)
        xy[j+1,alive] = next_xy[alive]
        valid2d[j+1] = alive
        xyz[j+1] = target3d
        projected[j+1], positive = project(target3d, k[j+1], np.eye(4))
        valid3d[j+1] = alive & positive & (source_mask[:,0]>=0.999)
        world[j+1] = transform_points(target3d, poses[j+1])
        finite_prob = np.isfinite(probability).all(1)
        # Float64 bilinear weights can sum to 1 + 2e-16 even for an
        # all-ones probability map. Tolerate rounding only, not invalid scores.
        if ((probability[finite_prob]<-1e-12) | (probability[finite_prob]>1+1e-12)).any():
            raise ValueError('Visibility/confidence must be probabilities')
        probability=np.clip(probability,0.,1.)
        ok = alive & finite_prob
        visconf[j+1,ok] = np.prod(probability[ok], axis=-1)
        valid3d[j+1] &= finite_prob
    projected[~valid3d] = np.nan
    xyz[~valid3d] = np.nan
    world[~valid3d] = np.nan
    return dict(xy_2d=xy, xy_projected=projected, xyz_camera=xyz,
                xyz_window_world=world, valid_2d=valid2d, valid_3d=valid3d,
                visconf=visconf, camera_to_world=poses.copy(), intrinsics_pixels=k)
