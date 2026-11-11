"""Validation shared by real cache production and calibrator training."""
import re
import numpy as np

ARRAY_NAMES = ('features', 'visual_logits', 'candidates', 'candidate_visibility',
               'valid', 'target_xy', 'target_visible', 'target_valid')


def validate_provenance(provenance):
    for name in ('track_on', 'verifier', 'track4world'):
        entry = provenance.get(name, {})
        if not re.fullmatch(r'[0-9a-f]{40}', entry.get('commit', '')):
            raise ValueError('Invalid baseline commit: ' + name)
        if not re.fullmatch(r'[0-9a-f]{64}', entry.get('weights_sha256', '')):
            raise ValueError('Invalid baseline weight hash: ' + name)
    tracker = provenance['track_on']
    expected = {'trackon2_dinov2': 'dinov2_s', 'trackon_r_dinov3': 'dinov3_s_plus'}
    variant = tracker.get('variant')
    if variant not in expected or tracker.get('backbone') != expected[variant]:
        raise ValueError('Tracker variant/backbone must be explicitly matched')
    if not re.fullmatch(r'[0-9a-f]{64}', tracker.get('backbone_weights_sha256', '')):
        raise ValueError('Missing frozen backbone weight hash')
    if not re.fullmatch(r'[0-9a-f]{64}', tracker.get('config_sha256', '')):
        raise ValueError('Missing tracker configuration hash')


def validate_arrays(arrays, feature_count):
    missing = set(ARRAY_NAMES) - set(arrays)
    if missing:
        raise ValueError('Missing cache arrays: ' + ', '.join(sorted(missing)))
    a = {k: np.asarray(arrays[k]) for k in ARRAY_NAMES}
    candidates = a['candidates']
    if candidates.ndim != 4 or candidates.shape[-1] != 2:
        raise ValueError('candidates must have shape [T,N,M,2]')
    t, n, m, _ = candidates.shape
    if min(t, n, m) < 1:
        raise ValueError('Empty cache dimensions')
    shapes = dict(features=(t,n,m,feature_count), visual_logits=(t,n,m),
                  candidates=(t,n,m,2), candidate_visibility=(t,n,m),
                  valid=(t,n,m), target_xy=(t,n,2), target_visible=(t,n),
                  target_valid=(t,n))
    for key, shape in shapes.items():
        if a[key].shape != shape:
            raise ValueError('Cache shape mismatch: ' + key)
        if a[key].dtype.kind not in 'biuf':
            raise ValueError('Non-numeric cache array: ' + key)
    for key in ('valid', 'target_visible', 'target_valid'):
        if not np.isin(a[key], [0,1]).all():
            raise ValueError('Mask is not binary: ' + key)
        a[key] = a[key].astype(bool)
    valid = a['valid']
    if not valid.any(-1).all():
        raise ValueError('Every frame/query needs at least one valid candidate')
    for key in ('features', 'visual_logits', 'candidates', 'candidate_visibility'):
        if not np.isfinite(a[key][valid]).all():
            raise ValueError('Non-finite valid candidate data: ' + key)
    visibility = a['candidate_visibility'][valid]
    if ((visibility < 0) | (visibility > 1)).any():
        raise ValueError('Candidate visibility must be a probability')
    spatial = a['target_valid'] & a['target_visible']
    if not np.isfinite(a['target_xy'][spatial]).all():
        raise ValueError('Visible labeled points require finite coordinates')
    # Invalid/occluded positions may be NaN; never turn them into supervision.
    return a
