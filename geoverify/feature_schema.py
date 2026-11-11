"""Shared feature order; missing nonnegative evidence uses -1, never fake zero."""
FEATURE_NAMES = (
    'visual_logit', 'trackon_visibility', 'geometry_visibility',
    'distance_trackon', 'distance_geometry_2d', 'distance_geometry_projected',
    'reprojection_residual', 'depth_change', 'velocity_residual',
    'acceleration_residual', 'neighborhood_residual', 'geometry_age', 'geometry_valid',
)
