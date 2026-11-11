"""Bounded one-job IPC with an independently installed Track4World runtime.

No frame is sent before capture. Slow jobs do not grow an unbounded queue.
Returned results carry capture and completion times in the caller's clock.
"""
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import subprocess
import tempfile
import time
import uuid
import numpy as np
from .interpreter import interpreter_path
from geoverify.geometry_worker import atomic_json


@dataclass(frozen=True)
class GeometryResult:
    job_id: str
    anchor_frame: int
    target_frame: int
    target_timestamp: float
    completed_timestamp: float
    query_ids: np.ndarray
    arrays: dict


def validate_window(rgb, queries, query_ids, frame_ids, timestamps, now):
    rgb, queries = np.asarray(rgb),np.asarray(queries)
    query_ids, frame_ids = np.asarray(query_ids),np.asarray(frame_ids)
    timestamps = np.asarray(timestamps,float)
    if rgb.ndim!=4 or rgb.shape[-1]!=3 or rgb.dtype!=np.uint8 or not 2<=len(rgb)<=16:
        raise ValueError('Expected 2..16 RGB uint8 frames')
    if frame_ids.shape!=(len(rgb),) or timestamps.shape!=(len(rgb),):
        raise ValueError('Frame metadata length mismatch')
    if frame_ids.dtype.kind not in 'iu' or (frame_ids<0).any() or (np.diff(frame_ids)<=0).any():
        raise ValueError('Frame IDs must strictly increase')
    if not math.isfinite(now) or not np.isfinite(timestamps).all():
        raise ValueError('Timestamps must be finite')
    if (np.diff(timestamps)<=0).any() or timestamps[-1]>now:
        raise ValueError('Window includes future or unordered frames')
    if queries.ndim!=2 or queries.shape[1]!=2 or not len(queries) or not np.isfinite(queries).all():
        raise ValueError('Queries must be finite nonempty [N,2]')
    if query_ids.shape!=(len(queries),) or query_ids.dtype.kind not in 'iu' or len(np.unique(query_ids))!=len(query_ids):
        raise ValueError('Query IDs must be unique integers')


class AsyncGeometryWorker:
    def __init__(self, python, root, checkpoint, ipc_root, physical_gpu,backbone_directory=None,clock=None,minimum_lag_frames=0):
        if type(minimum_lag_frames) is not int or not 0<=minimum_lag_frames<256:
            raise ValueError('minimum_lag_frames must be an integer in [0,255]')
        self.minimum_lag_frames=minimum_lag_frames
        # No global CUDA environment mutation in the Track-On parent process.
        python=interpreter_path(python)
        root, checkpoint = (Path(p).resolve() for p in (root,checkpoint))
        if not python.is_file() or not checkpoint.is_file():
            raise FileNotFoundError('Worker interpreter and checkpoint must exist')
        ipc_root = Path(ipc_root).resolve()
        ipc_root.mkdir(parents=True,exist_ok=True)
        self.directory = Path(tempfile.mkdtemp(prefix='geometry-',dir=ipc_root))
        project = Path(__file__).resolve().parent.parent
        environment = os.environ.copy()
        environment['CUDA_VISIBLE_DEVICES'] = str(physical_gpu)
        environment['PYTHONPATH'] = str(project)
        self.log = (self.directory/'worker.log').open('wb')
        backbone_args=[] if backbone_directory is None else ['--backbone-directory',str(Path(backbone_directory).resolve())]
        self.process = subprocess.Popen(
            [str(python),'-m','geoverify.geometry_worker','--directory',str(self.directory),
             '--root',str(root),'--checkpoint',str(checkpoint)]+backbone_args,
            cwd=str(project),env=environment,stdout=self.log,stderr=subprocess.STDOUT)
        self.pending = None
        self.last_target = -1
        self.last_poll_time = -math.inf
        self.closed = False
        self.clock = clock

    def _check(self):
        if self.closed:
            raise RuntimeError('Worker is closed')
        error = self.directory/'error.json'
        if error.exists():
            raise RuntimeError(json.loads(error.read_text())['error'])
        if self.process.poll() is not None:
            raise RuntimeError('Geometry worker exited; inspect '+str(self.directory/'worker.log'))

    @property
    def ready(self):
        self._check()
        return (self.directory/'ready.json').exists()

    def wait_ready(self,timeout=600.,on_wait=None):
        """Wait before capture starts, never run a video with an unloaded model."""
        if not math.isfinite(timeout) or timeout<=0:
            raise ValueError('Positive finite initialization timeout required')
        start=time.monotonic()
        last_report=start
        while not self.ready:
            now=time.monotonic()
            if now-start>=timeout:
                raise TimeoutError('Geometry initialization timed out; inspect '+str(self.directory/'worker.log'))
            if on_wait is not None and now-last_report>=10:
                on_wait(now-start,str(self.directory/'worker.log'))
                last_report=now
            time.sleep(min(.1,timeout-(now-start)))

    def submit(self,rgb,queries,query_ids,frame_ids,timestamps,now):
        """False means busy/not initialized, not a substituted geometry result."""
        validate_window(rgb,queries,query_ids,frame_ids,timestamps,now)
        self._check()
        if self.pending is not None or not self.ready:
            return False
        if int(frame_ids[-1])<=self.last_target:
            raise ValueError('Cannot submit an older geometry window')
        if now<self.last_poll_time:
            raise ValueError('Caller clock moved backwards')
        submitted_monotonic=time.monotonic()
        if getattr(self,'clock',None) is not None:
            observed=float(self.clock())
            if not math.isfinite(observed) or observed<now:
                raise ValueError('Capture clock moved backwards during submission')
            now=observed
        job = dict(job_id=uuid.uuid4().hex,anchor_frame=int(frame_ids[0]),
                   target_frame=int(frame_ids[-1]),target_timestamp=float(timestamps[-1]),
                   submitted_timestamp=float(now),submitted_monotonic=submitted_monotonic,
                   query_ids=np.array(query_ids,copy=True))
        # Previous job was consumed before reusing these fixed-size IPC files.
        np.save(self.directory/'frames.npy',rgb,allow_pickle=False)
        np.save(self.directory/'queries.npy',queries,allow_pickle=False)
        atomic_json(self.directory/'request.json',{k:v for k,v in job.items() if k!='query_ids'})
        self.pending = job
        self.last_target = job['target_frame']
        return True

    def poll(self,current_frame,now):
        self._check()
        if not math.isfinite(now) or now<self.last_poll_time:
            raise ValueError('Caller clock must be finite and monotonic')
        self.last_poll_time = now
        response_path = self.directory/'response.json'
        if self.pending is None or not response_path.exists():
            return None
        response = json.loads(response_path.read_text())
        job = self.pending
        if response['job_id']!=job['job_id']:
            return None  # Previous completed response, pending job still running.
        if response['target_frame']!=job['target_frame'] or response['anchor_frame']!=job['anchor_frame']:
            raise ValueError('Worker returned mismatched frame metadata')
        elapsed = response['completed_monotonic']-job['submitted_monotonic']
        if not math.isfinite(elapsed) or elapsed<0:
            raise ValueError('Invalid worker completion clock')
        completed = job['submitted_timestamp']+elapsed
        if current_frame<job['target_frame']:
            raise ValueError('Geometry references a future frame')
        # Controlled additional delivery lag for TRAIN robustness experiments.
        # Never fabricate completion, replace outputs, or clear pending early.
        if current_frame-job['target_frame']<getattr(self,'minimum_lag_frames',0):
            return None
        if now<completed:
            return None  # Offline playback cannot publish before logical availability.
        with np.load(self.directory/'result.npz',allow_pickle=False) as result:
            arrays = {k:result[k] for k in result.files}
        self.pending = None
        return GeometryResult(job['job_id'],job['anchor_frame'],job['target_frame'],
                              job['target_timestamp'],completed,job['query_ids'],arrays)

    def close(self):
        if self.closed:
            return
        (self.directory/'stop').touch()
        try:
            self.process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.process.terminate()  # Only this instance's owned child process.
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
        self.log.close()
        self.closed = True
        # Preserve logs and artifacts for diagnostics; no recursive deletion.
