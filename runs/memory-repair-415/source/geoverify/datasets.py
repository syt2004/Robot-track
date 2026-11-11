"""Complete-manifest robot-video readers; labels stay outside model interfaces."""
from dataclasses import dataclass
import hashlib
import io
import json
from pathlib import Path
import pickle
import numpy as np


def file_sha256(path):
    value=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):
            value.update(block)
    return value.hexdigest()


class NumpyDatasetUnpickler(pickle.Unpickler):
    """Accept only numpy's array serialization primitives, not arbitrary globals."""
    def find_class(self,module,name):
        # RoboTAP stores mediapy's ndarray subclass. Its inherited ndarray
        # pickle state is plain array data; do not import arbitrary classes.
        if module=='mediapy' and name=='_VideoArray':
            return np.ndarray
        if module=='numpy' and name in ('ndarray','dtype'):
            return getattr(np,name)
        if module in ('numpy.core.multiarray','numpy._core.multiarray') and name in ('_reconstruct','scalar'):
            return getattr(np.core.multiarray,name)
        if module in ('numpy.core.numeric','numpy._core.numeric') and name=='_frombuffer':
            return np.core.numeric._frombuffer
        raise pickle.UnpicklingError('Unsupported dataset pickle global: '+module+'.'+name)


@dataclass
class VideoSample:
    name: str
    split: str
    rgb: np.ndarray
    timestamps: np.ndarray
    query_txy: np.ndarray
    target_xy: np.ndarray
    target_visible: np.ndarray
    source_sha256: str


def robotap_record(name,split,record,fps,source_sha256):
    if not np.isfinite(fps) or fps<=0:
        raise ValueError('Explicit positive source/assumed FPS required')
    frames=record['video']
    if len(frames)==0:
        raise ValueError('Empty robot video')
    if isinstance(frames[0],bytes):
        from PIL import Image
        frames=np.stack([np.asarray(Image.open(io.BytesIO(frame)).convert('RGB')) for frame in frames])
    frames=np.asarray(frames)
    if frames.ndim!=4 or frames.shape[-1]!=3 or frames.dtype!=np.uint8:
        raise ValueError('Dataset video must decode to RGB uint8 THWC')
    points=np.asarray(record['points'],float)
    occluded=np.asarray(record['occluded'])
    if points.ndim!=3 or points.shape[1:]!=(len(frames),2) or occluded.shape!=points.shape[:2]:
        raise ValueError('Dataset point/video shape mismatch')
    if not np.isin(occluded,[0,1]).all():
        raise ValueError('Dataset occlusion must be binary')
    visible=~occluded.astype(bool)
    if not np.isfinite(points[visible]).all():
        raise ValueError('Non-finite visible ground truth')
    # Match TAP-Vid first-query protocol: omit tracks never visible, nothing else.
    keep=visible.any(-1)
    points,visible=points[keep],visible[keep]
    if not len(points):
        raise ValueError('No queryable point in video')
    points=points*np.array([frames.shape[2],frames.shape[1]])
    first=visible.argmax(-1)
    queries=np.c_[first,points[np.arange(len(points)),first]]  # t,x,y, not t,y,x
    return VideoSample(str(name),split,frames,np.arange(len(frames))/fps,queries,
                       points.transpose(1,0,2),visible.T,source_sha256)


def iter_robotap(manifest_path,split):
    """Read every listed file/video for the requested split; never truncate clips.

    Manifest schema 1: dataset='robotap', files=[path,sha256,split,fps].
    FPS must be explicitly documented by the manifest author; a pickle does not
    contain reliable container timestamps. No inferred real-time claims are made.
    """
    manifest_path=Path(manifest_path).resolve()
    manifest=json.loads(manifest_path.read_text())
    if manifest.get('schema_version')!=1 or manifest.get('dataset') not in ('robotap','tapvid_davis','tapvid_rgb_stacking'):
        raise ValueError('Expected TAP-Vid/RoboTAP manifest schema 1')
    selected=[f for f in manifest['files'] if f['split']==split]
    if not selected:
        raise ValueError('Requested split has no files')
    names=set()
    paths=set()
    for entry in selected:
        path=(manifest_path.parent/entry['path']).resolve()
        if path in paths:
            raise ValueError('Duplicate dataset shard')
        paths.add(path)
        if file_sha256(path)!=entry['sha256']:
            raise ValueError('Dataset checksum mismatch: '+str(path))
        with path.open('rb') as stream:
            records=NumpyDatasetUnpickler(stream).load()
        if not isinstance(records,dict) or not records:
            raise ValueError('Expected nonempty dictionary of RoboTAP videos')
        for name in sorted(records):
            if name in names:
                raise ValueError('Duplicate video name across shards')
            names.add(name)
            yield robotap_record(name,split,records[name],entry['fps'],entry['sha256'])


def iter_evaluation(manifest_path,split):
    """Dispatch only explicit dataset schemas, preserving each coordinate convention."""
    manifest=json.loads(Path(manifest_path).read_text())
    if manifest.get('dataset')=='cotracker3_kubric':
        from .kubric import iter_kubric
        yield from iter_kubric(manifest_path,split)
    else:
        yield from iter_robotap(manifest_path,split)
