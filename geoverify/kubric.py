"""Full-sequence CoTracker3 Kubric MOVi-F supervision, matching Track-On loader.

The source field named 'visibility' is inverted by the official Track-On
loader: True there means occluded. Coordinates are already pixel x/y.
No temporal cropping, random scene replacement or implicit point subsampling.
"""
import json
from pathlib import Path
import numpy as np
from .datasets import NumpyDatasetUnpickler,VideoSample,file_sha256


def read_kubric_annotations(path):
    with Path(path).open('rb') as stream:
        version=np.lib.format.read_magic(stream)
        if version==(1,0):
            shape,_,dtype=np.lib.format.read_array_header_1_0(stream)
        elif version==(2,0):
            shape,_,dtype=np.lib.format.read_array_header_2_0(stream)
        else:
            raise ValueError('Unsupported Kubric NPY header version')
        if shape!=() or not dtype.hasobject:
            raise ValueError('Expected scalar numpy annotation dictionary')
        value=NumpyDatasetUnpickler(stream).load()
        if not isinstance(value,np.ndarray) or value.shape!=():
            raise ValueError('Invalid annotation payload')
        record=value.item()
        if not isinstance(record,dict) or not {'coords','visibility'}<=set(record):
            raise ValueError('Missing Kubric trajectory annotation fields')
        return record


def kubric_sample(name,split,rgb,annotation,fps,source_sha256):
    rgb=np.asarray(rgb)
    points=np.asarray(annotation['coords'],float)
    occluded=np.asarray(annotation['visibility'])
    if rgb.ndim!=4 or rgb.shape[-1]!=3 or rgb.dtype!=np.uint8 or not len(rgb):
        raise ValueError('Kubric RGB must be nonempty uint8 THWC')
    if not np.isfinite(fps) or fps<=0:
        raise ValueError('Explicit positive FPS required')
    if points.ndim!=3 or points.shape[1:]!=(len(rgb),2) or occluded.shape!=points.shape[:2]:
        raise ValueError('Kubric trajectory/frame shape mismatch')
    if not np.isin(occluded,[0,1]).all():
        raise ValueError('Kubric visibility field must be binary occlusion flags')
    visible=~occluded.astype(bool)
    if not np.isfinite(points[visible]).all():
        raise ValueError('Non-finite visible Kubric point')
    h,w=rgb.shape[1:3]
    visible &= (points[...,0]>=0)&(points[...,0]<w)&(points[...,1]>=0)&(points[...,1]<h)
    keep=visible.any(-1)
    points,visible=points[keep],visible[keep]
    if not len(points):
        raise ValueError('No queryable point; scene is not silently replaced')
    first=visible.argmax(-1)
    query=np.c_[first,points[np.arange(len(points)),first]]
    return VideoSample(str(name),split,rgb,np.arange(len(rgb))/fps,query,
                       points.transpose(1,0,2),visible.T,source_sha256)


def iter_kubric(manifest_path,split):
    """Schema 1 dataset=cotracker3_kubric, scenes with locked annotations/frames.

    Each scene: name, split, fps, annotation={path,sha256},
    frames=[{path,sha256}, ...] in chronological order. The manifest creator must
    enumerate the full selected source split; no wildcard sampling here.
    """
    from PIL import Image
    manifest_path=Path(manifest_path).resolve()
    manifest=json.loads(manifest_path.read_text())
    if manifest.get('schema_version')!=1 or manifest.get('dataset')!='cotracker3_kubric':
        raise ValueError('Expected CoTracker3 Kubric manifest schema 1')
    scenes=[s for s in manifest['scenes'] if s['split']==split]
    if not scenes or len({s['name'] for s in scenes})!=len(scenes):
        raise ValueError('Empty split or duplicate scene name')
    annotation_paths=set()
    for scene in scenes:
        def checked(entry):
            path=(manifest_path.parent/entry['path']).resolve()
            if file_sha256(path)!=entry['sha256']:
                raise ValueError('Kubric checksum mismatch: '+str(path))
            return path
        annotation_path=checked(scene['annotation'])
        if annotation_path in annotation_paths:
            raise ValueError('Duplicate annotation file')
        annotation_paths.add(annotation_path)
        frame_paths=[checked(entry) for entry in scene['frames']]
        if not frame_paths or len(set(frame_paths))!=len(frame_paths):
            raise ValueError('Missing or duplicate RGB frames')
        frames=[]
        for path in frame_paths:
            with Image.open(path) as image:
                frames.append(np.asarray(image.convert('RGB')))
        sample=kubric_sample(scene['name'],split,np.stack(frames),read_kubric_annotations(annotation_path),
                             scene['fps'],scene['annotation']['sha256'])
        if 'query_indices' in scene:
            indices=np.asarray(scene['query_indices'])
            if (indices.ndim!=1 or not len(indices) or indices.dtype.kind not in 'iu'
                    or len(np.unique(indices))!=len(indices)
                    or ((indices<0)|(indices>=len(sample.query_txy))).any()
                    or not manifest.get('query_selection')):
                raise ValueError('Explicit unique query subset and selection provenance required')
            sample=VideoSample(sample.name,sample.split,sample.rgb,sample.timestamps,
                               sample.query_txy[indices],sample.target_xy[:,indices],
                               sample.target_visible[:,indices],sample.source_sha256)
        yield sample
