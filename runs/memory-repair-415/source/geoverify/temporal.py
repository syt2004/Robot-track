"""Bounded, timestamp-aware correction state without rewriting emitted history."""
from dataclasses import dataclass
import math
import numpy as np


@dataclass(frozen=True)
class Correction:
    anchor_frame: int
    target_frame: int
    target_timestamp: float
    completed_timestamp: float
    delta_xy: np.ndarray
    valid: np.ndarray
    window_id: str


class CorrectionBuffer:
    def __init__(self, tau_seconds=0.25, max_age_seconds=1.0):
        if not math.isfinite(tau_seconds) or not math.isfinite(max_age_seconds) or tau_seconds <= 0 or max_age_seconds <= 0:
            raise ValueError("Positive time constants required")
        self.tau = tau_seconds
        self.max_age = max_age_seconds
        self.latest = None

    def accept(self, correction, current_frame, current_timestamp):
        if not all(math.isfinite(t) for t in (correction.target_timestamp,
                                              correction.completed_timestamp,current_timestamp)):
            raise ValueError('Correction times must be finite')
        if not (0 <= correction.anchor_frame <= correction.target_frame <= current_frame):
            raise ValueError("Geometry references an invalid or future frame")
        if not (correction.target_timestamp <= correction.completed_timestamp <= current_timestamp):
            raise ValueError("Correction is not yet available")
        delta = np.asarray(correction.delta_xy)
        valid = np.asarray(correction.valid)
        if delta.ndim != 2 or delta.shape[1] != 2 or valid.shape != (len(delta),):
            raise ValueError("Correction shapes must be [N,2] and [N]")
        if not np.isin(valid,[0,1]).all():
            raise ValueError('Correction validity must be binary')
        valid = valid.astype(bool)
        if not np.isfinite(delta[valid]).all():
            raise ValueError("Valid correction contains NaN/Inf")
        if self.latest is not None and correction.target_frame <= self.latest.target_frame:
            return False
        # A frozen dataclass does not make caller-owned numpy arrays immutable.
        self.latest = Correction(correction.anchor_frame,correction.target_frame,
                                 correction.target_timestamp,correction.completed_timestamp,
                                 delta.copy(),valid.copy(),correction.window_id)
        return True

    def apply(self, trackon_xy, timestamp, evidence_timestamp=None):
        result = np.asarray(trackon_xy, dtype=float).copy()
        if result.ndim!=2 or result.shape[1]!=2 or not np.isfinite(result).all() or not math.isfinite(timestamp):
            raise ValueError('Finite timestamp and [N,2] positions required')
        applied = np.zeros(len(result), dtype=bool)
        if self.latest is None:
            return result, applied
        c = self.latest
        if timestamp < c.completed_timestamp:
            raise ValueError("Cannot use result before completion")
        reference = timestamp if evidence_timestamp is None else float(evidence_timestamp)
        if not math.isfinite(reference) or not c.target_timestamp<=reference<=timestamp:
            raise ValueError('Invalid evidence age reference')
        age = reference-c.target_timestamp
        if age > self.max_age:
            return result, applied
        if result.shape != c.delta_xy.shape:
            raise ValueError("Query identity/count changed; reset correction state")
        applied = np.asarray(c.valid, bool).copy()
        result[applied] += math.exp(-age/self.tau)*c.delta_xy[applied]
        return result, applied
