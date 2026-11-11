"""Geometry conventions: row points, column transforms, world-to-camera extrinsics.

No function silently joins independent reconstruction window coordinate systems.
"""
import numpy as np


def transform_points(points, transform):
    points = np.asarray(points, dtype=np.float64)
    transform = np.asarray(transform, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or transform.shape != (4, 4):
        raise ValueError("Expected points [N,3] and transform [4,4]")
    return points @ transform[:3, :3].T + transform[:3, 3]


def project(points_world, intrinsics, world_to_camera):
    """Return pixels and positive-depth validity; invalid pixels are NaN."""
    camera = transform_points(points_world, world_to_camera)
    k = np.asarray(intrinsics, dtype=np.float64)
    if k.shape != (3, 3):
        raise ValueError("Intrinsics must be [3,3]")
    valid = np.isfinite(camera).all(axis=1) & (camera[:, 2] > 1e-8)
    uv = np.full((len(camera), 2), np.nan)
    homogeneous = camera[valid] @ k.T
    uv[valid] = homogeneous[:, :2] / homogeneous[:, 2:]
    return uv, valid


def bilinear_sample(field, xy):
    """Sample HWC data at integer-center pixel coordinates; do not clamp invalids."""
    field = np.asarray(field)
    xy = np.asarray(xy, dtype=np.float64)
    if field.ndim != 3 or xy.ndim != 2 or xy.shape[1] != 2:
        raise ValueError("Expected [H,W,C] field and [N,2] xy")
    h, w, c = field.shape
    valid = np.isfinite(xy).all(axis=1)
    valid &= (xy[:, 0] >= 0) & (xy[:, 0] <= w-1)
    valid &= (xy[:, 1] >= 0) & (xy[:, 1] <= h-1)
    result = np.full((len(xy), c), np.nan)
    x, y = xy[valid].T
    x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
    x1, y1 = np.minimum(x0+1, w-1), np.minimum(y0+1, h-1)
    ax, ay = (x-x0)[:, None], (y-y0)[:, None]
    result[valid] = ((1-ax)*(1-ay)*field[y0,x0] + ax*(1-ay)*field[y0,x1]
                     + (1-ax)*ay*field[y1,x0] + ax*ay*field[y1,x1])
    return result, valid


def weighted_kabsch(source, target, weights):
    """Fit target = R source + t. Reject collinear or insufficient support."""
    a, b = np.asarray(source, float), np.asarray(target, float)
    weights = np.asarray(weights, float)
    if a.ndim != 2 or a.shape[1] != 3 or b.shape != a.shape or weights.shape != (len(a),):
        raise ValueError("Expected paired [N,3] clouds and [N] weights")
    valid = np.isfinite(a).all(1) & np.isfinite(b).all(1) & np.isfinite(weights) & (weights > 0)
    a, b, weights = a[valid], b[valid], weights[valid]
    if len(a) < 3:
        raise ValueError("At least three valid correspondences required")
    weights = weights / weights.sum()
    ca, cb = weights @ a, weights @ b
    ac, bc = a-ca, b-cb
    u, singular, vt = np.linalg.svd((ac * weights[:, None]).T @ bc)
    if singular[0] <= 1e-12 or singular[1] <= singular[0]*1e-8:
        raise ValueError("Degenerate (collinear) correspondence geometry")
    correction = np.eye(3)
    correction[-1,-1] = np.linalg.det(vt.T @ u.T)
    rotation = vt.T @ correction @ u.T
    transform = np.eye(4)
    transform[:3,:3] = rotation
    transform[:3,3] = cb-rotation@ca
    return transform


def robust_rigid_fit(source, target, weights, threshold, iterations=128, seed=0):
    """Weighted sampling/scoring followed by inlier refit. Threshold uses input units."""
    a, b, w = np.asarray(source,float), np.asarray(target,float), np.asarray(weights,float)
    if threshold <= 0 or iterations < 1:
        raise ValueError("Positive threshold and iteration count required")
    valid = np.isfinite(a).all(1) & np.isfinite(b).all(1) & np.isfinite(w) & (w>0)
    indices = np.flatnonzero(valid)
    if len(indices) < 3:
        raise ValueError("Insufficient valid points")
    probability = w[indices] / w[indices].sum()
    rng = np.random.default_rng(seed)
    best, best_score = None, -1.
    for _ in range(iterations):
        sample = rng.choice(indices, 3, replace=False, p=probability)
        try:
            pose = weighted_kabsch(a[sample], b[sample], w[sample])
        except ValueError:
            continue
        residual = np.linalg.norm(transform_points(a,pose)-b, axis=1)
        inliers = valid & (residual <= threshold)
        score = w[inliers].sum()
        if inliers.sum() >= 3 and score > best_score:
            best, best_score = inliers, score
    if best is None:
        raise ValueError("No non-degenerate rigid consensus")
    pose = weighted_kabsch(a[best], b[best], w[best])
    residual = np.linalg.norm(transform_points(a,pose)-b, axis=1)
    final = valid & (residual <= threshold)
    return pose, final, residual
