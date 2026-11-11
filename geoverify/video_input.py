"""Decode every video frame and validate user queries without ground truth."""
import json
from pathlib import Path
import subprocess
import numpy as np


def decode_video(path):
    path=Path(path).resolve(strict=True)
    result=subprocess.run(['ffprobe','-v','error','-select_streams','v:0',
                           '-show_entries','stream=width,height:frame=best_effort_timestamp_time',
                           '-of','json',str(path)],capture_output=True,text=True,check=True)
    metadata=json.loads(result.stdout)
    stream=metadata['streams'][0]
    w,h=int(stream['width']),int(stream['height'])
    timestamps=np.asarray([float(frame['best_effort_timestamp_time'])
                           for frame in metadata['frames']],float)
    if len(timestamps)<2 or not np.isfinite(timestamps).all() or (np.diff(timestamps)<=0).any():
        raise ValueError('Video needs at least two frames with increasing presentation timestamps')
    timestamps-=timestamps[0]
    # Never silently resample or drop input frames to satisfy the MP4 renderer.
    dt=np.diff(timestamps)
    if not np.allclose(dt,dt.mean(),rtol=1e-4,atol=2e-6):
        raise ValueError('Variable-frame-rate input requires explicit conversion before this demo')
    # ffprobe prints timestamps rounded to microseconds; use the measured duration
    # to remove only rounding jitter, not meaningful cadence changes.
    timestamps=np.linspace(0,timestamps[-1],len(timestamps))
    decoded=subprocess.run(['ffmpeg','-v','error','-noautorotate','-i',str(path),
                            '-map','0:v:0','-vsync','0','-f','rawvideo','-pix_fmt','rgb24','pipe:1'],
                           capture_output=True,check=True)
    expected=len(timestamps)*h*w*3
    if len(decoded.stdout)!=expected:
        raise ValueError('Decoded frame count differs from probed presentation timestamps')
    return np.frombuffer(decoded.stdout,np.uint8).reshape(-1,h,w,3),timestamps


def read_queries(path,video_shape):
    """Schema 1: query_txy is [frame index, x pixel, y pixel], not supervision."""
    value=json.loads(Path(path).read_text())
    if value.get('schema_version')!=1:
        raise ValueError('Query JSON requires schema_version 1')
    query=np.asarray(value['query_txy'],float)
    t,h,w,_=video_shape
    if query.ndim!=2 or query.shape[1]!=3 or not len(query) or not np.isfinite(query).all():
        raise ValueError('Queries must be a finite nonempty [N,3] array')
    if ((query[:,0]<0)|(query[:,0]>=t)|(query[:,0]!=np.floor(query[:,0]))).any():
        raise ValueError('Query frame index outside video')
    if ((query[:,1]<0)|(query[:,1]>=w)|(query[:,2]<0)|(query[:,2]>=h)).any():
        raise ValueError('Query pixel outside unrotated decoded image')
    objects=value.get('object_ids')
    if objects is not None:
        objects=np.asarray(objects)
        if objects.shape!=(len(query),) or objects.dtype.kind not in 'iu':
            raise ValueError('object_ids must contain one integer per query')
    return query,objects


def mask_queries(path,video_shape,max_points_per_object=64):
    """Sample spatial cells of a first-frame integer mask; zero is background.

    Each positive label is an externally specified object, not an inferred GT
    trajectory group. Thin objects retain representatives even off a regular grid.
    """
    from PIL import Image
    if isinstance(max_points_per_object,bool) or not isinstance(max_points_per_object,int) or max_points_per_object<1:
        raise ValueError('Positive integer max_points_per_object required')
    with Image.open(path) as image:
        mask=np.asarray(image)
    if mask.shape!=tuple(video_shape[1:3]) or mask.dtype.kind not in 'biu' or (mask<0).any():
        raise ValueError('Mask must be a single-channel nonnegative integer image at decoded resolution')
    queries,objects=[],[]
    for label in np.unique(mask):
        if label==0:
            continue
        y,x=np.nonzero(mask==label)
        stride=max(1,int(np.sqrt(len(x)/max_points_per_object)))
        cx,cy=x//stride,y//stride
        cell=cy*(mask.shape[1]//stride+1)+cx
        distance=(x-(cx+.5)*stride)**2+(y-(cy+.5)*stride)**2
        order=np.lexsort((x,y,distance,cell))
        ordered_cells=cell[order]
        representatives=order[np.r_[True,ordered_cells[1:]!=ordered_cells[:-1]]]
        if len(representatives)>max_points_per_object:
            representatives=representatives[np.linspace(0,len(representatives)-1,max_points_per_object,dtype=int)]
        for index in representatives:
            queries.append([0,int(x[index]),int(y[index])])
            objects.append(int(label))
    if not queries:
        raise ValueError('Mask has no foreground objects')
    return np.asarray(queries,float),np.asarray(objects,np.int64)
