"""Bounded provenance for predictor memory; no labels or geometry inference."""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class MemoryEdit:
    job_id: str
    query_id: int
    frame_id: int
    completed_timestamp: float


class MemoryLedger:
    """One ledger per cohort. Slots follow Predictor's left-roll write order."""

    def __init__(self, query_ids, memory_size):
        ids = np.asarray(query_ids)
        if ids.ndim != 1 or ids.dtype.kind not in 'iu' or len(np.unique(ids)) != len(ids):
            raise ValueError('Unique integer query IDs required')
        if memory_size < 1:
            raise ValueError('Positive memory size required')
        self.ids = ids.copy()
        self.rows = {int(q): i for i, q in enumerate(ids)}
        self.frames = np.full(memory_size, -1, dtype=np.int64)
        self.times = np.full(memory_size, np.nan)
        self.xy = np.zeros((len(ids), memory_size, 2))
        self.visibility = np.zeros((len(ids), memory_size))
        self.edited = [set() for _ in range(memory_size)]
        self.last_frame = -1
        self.last_time = -np.inf

    def append(self, frame_id, timestamp, xy, visibility):
        xy, visibility = np.asarray(xy), np.asarray(visibility)
        if (frame_id <= self.last_frame or not np.isfinite(timestamp)
                or timestamp <= self.last_time):
            raise ValueError('Frame and capture time must increase')
        if xy.shape != (len(self.ids), 2) or visibility.shape != (len(self.ids),):
            raise ValueError('Query shape mismatch')
        if not np.isfinite(xy).all() or not np.isfinite(visibility).all():
            raise ValueError('Finite predictions required')
        self.frames = np.roll(self.frames, -1)
        self.times = np.roll(self.times, -1)
        self.xy = np.roll(self.xy, -1, axis=1)
        self.visibility = np.roll(self.visibility, -1, axis=1)
        self.frames[-1], self.times[-1] = frame_id, timestamp
        self.xy[:, -1], self.visibility[:, -1] = xy, visibility
        self.edited = self.edited[1:] + [set()]
        self.last_frame, self.last_time = frame_id, timestamp

    def locate(self, edit, now):
        if (not np.isfinite(now) or not np.isfinite(edit.completed_timestamp)
                or edit.completed_timestamp > now):
            return None
        row = self.rows.get(edit.query_id)
        slots = np.flatnonzero(self.frames == edit.frame_id)
        if row is None or len(slots) != 1 or edit.frame_id < 0:
            return None
        slot = int(slots[0])
        if edit.completed_timestamp < self.times[slot]:
            return None
        if (edit.job_id, edit.query_id) in self.edited[slot]:
            return None
        return row, slot

    def commit(self, edit, now):
        """Call only after the corresponding predictor mutation succeeds."""
        location = self.locate(edit, now)
        if location is None:
            raise ValueError('Expired, duplicate, unavailable or unknown edit')
        self.edited[location[1]].add((edit.job_id, edit.query_id))
        return location
