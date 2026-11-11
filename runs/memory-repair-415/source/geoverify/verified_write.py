"""Compare historical geometry against an independent trajectory before state writes.

All images and positions are from the already received geometry window. No GT,
future images, repaired-state reference or revised historical outputs are used.
"""
import numpy as np
import torch
from .adapters import masked_candidate_attention
from .rgb_transport import RGBFloatTransport


def admission_from_scores(scores, valid):
    scores = np.asarray(scores, dtype=np.float32)
    valid = np.asarray(valid, dtype=bool)
    if scores.shape != (*valid.shape, 2):
        raise ValueError('Expected paired scores [time, query, reference/geometry]')
    margin = scores[..., 1] - scores[..., 0]
    finite = np.isfinite(scores).all(-1)
    # Numerical tie tolerance, not a threshold fitted on robot quality.
    accept = valid & finite & (margin > 1e-6)
    features = np.stack((np.tanh(np.where(finite, margin, 0)),
                         np.broadcast_to(accept.mean(0), valid.shape)), -1).astype(np.float32)
    return accept, features


def verify_historical_window(verifier, rgb, reference_xy, geometry_xy, valid):
    rgb = np.asarray(rgb, dtype=np.uint8)
    ref = np.asarray(reference_xy, dtype=np.float32)
    geo = np.asarray(geometry_xy, dtype=np.float32)
    valid = np.asarray(valid, dtype=bool).copy()
    if ref.shape != geo.shape or ref.shape[:2] != valid.shape or len(rgb) != len(ref):
        raise ValueError('Historical verification shape mismatch')
    if not np.isfinite(ref).all():
        raise ValueError('Independent reference must be finite')
    h, w = rgb.shape[1:3]
    valid &= np.isfinite(geo).all(-1)
    valid &= (geo[..., 0] >= 0) & (geo[..., 0] < w) & (geo[..., 1] >= 0) & (geo[..., 1] < h)
    tracks = np.stack((ref, np.where(valid[..., None], geo, ref)), axis=2)
    available = np.stack((np.ones_like(valid), valid), axis=2)
    query = np.c_[np.zeros(ref.shape[1]), ref[0]].astype(np.float32)
    if not hasattr(verifier, '_rgb_float_transport'):
        verifier._rgb_float_transport = RGBFloatTransport(verifier.device)
    transport = verifier._rgb_float_transport
    with torch.inference_mode():
        video = transport.prepare(rgb)
        track_tensor = torch.as_tensor(tracks, device=verifier.device)[None]
        mask = torch.as_tensor(available, device=verifier.device)[None]
        queries = torch.as_tensor(query, device=verifier.device)[None]
        with masked_candidate_attention(verifier.model, mask):
            logits = verifier.model.efficient_forward(video, track_tensor, queries)
        scores = logits[0].float().cpu().numpy()
        transport.finish()
    accept, features = admission_from_scores(scores, valid)
    return dict(scores=scores, accept=accept, features=features, valid=valid)
