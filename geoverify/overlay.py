"""Render full-length causal track overlays using Pillow and system ffmpeg."""
import colorsys
import json
from pathlib import Path
import subprocess
import tempfile
import numpy as np


def rigid_event_markers(arrays):
    """Mark delayed rigid inlier events at emission, never rewrite target frames."""
    shape=np.asarray(arrays['xy']).shape[:2]
    markers=np.zeros(shape,bool)
    events=json.loads(str(arrays.get('object_motion_json','[]')))
    for event in events:
        if event['status']!='observed':
            continue
        frame=event['emitted_frame']
        ids=np.asarray(event['query_ids'])
        inliers=np.asarray(event['inlier_mask'])
        if (not isinstance(frame,int) or not 0<=frame<shape[0]
                or ids.ndim!=1 or ids.dtype.kind not in 'iu'
                or ((ids<0)|(ids>=shape[1])).any() or len(np.unique(ids))!=len(ids)
                or inliers.shape!=ids.shape or not np.isin(inliers,[0,1]).all()
                or event['target_frame']>frame):
            raise ValueError('Invalid rigid overlay event')
        markers[frame,ids[inliers.astype(bool)]]=True
    return markers


def render_overlay(rgb,timestamps,xy,visible,confidence,path,label,trail=20,rigid_inliers=None):
    from PIL import Image,ImageDraw
    rgb,xy=np.asarray(rgb),np.asarray(xy)
    visible,confidence=np.asarray(visible),np.asarray(confidence)
    timestamps=np.asarray(timestamps,float)
    if rgb.ndim!=4 or rgb.dtype!=np.uint8 or rgb.shape[-1]!=3 or len(rgb)<2:
        raise ValueError('Overlay requires at least two RGB uint8 frames')
    if xy.ndim!=3 or xy.shape[0]!=len(rgb) or xy.shape[-1]!=2:
        raise ValueError('Overlay track/video shape mismatch')
    if visible.shape!=xy.shape[:2] or confidence.shape!=visible.shape or not np.isin(visible,[0,1]).all():
        raise ValueError('Overlay visibility/confidence shape mismatch')
    if not np.isfinite(xy).all() or not np.isfinite(confidence).all() or ((confidence<0)|(confidence>1)).any():
        raise ValueError('Invalid overlay values')
    if rigid_inliers is not None:
        rigid_inliers=np.asarray(rigid_inliers)
        if rigid_inliers.shape!=visible.shape or not np.isin(rigid_inliers,[0,1]).all():
            raise ValueError('Invalid rigid inlier marker mask')
    if timestamps.shape!=(len(rgb),) or not np.isfinite(timestamps).all():
        raise ValueError('Invalid overlay timestamps')
    dt=np.diff(timestamps)
    if (dt<=0).any() or not np.allclose(dt,dt[0],rtol=1e-6,atol=1e-9):
        raise ValueError('Constant-frame-rate export requires uniformly spaced timestamps')
    if not isinstance(trail,int) or trail<1:
        raise ValueError('Positive trail length required')
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    h,w=rgb.shape[1:3]
    colors=[tuple(int(c*255) for c in colorsys.hsv_to_rgb((i*.61803398875)%1,.85,1.)) for i in range(xy.shape[1])]
    command=['ffmpeg','-hide_banner','-loglevel','error','-n','-f','rawvideo','-pix_fmt','rgb24',
             '-s','%dx%d'%(w,h),'-r',str(1./dt[0]),'-i','pipe:0','-an',
             '-vf','pad=ceil(iw/2)*2:ceil(ih/2)*2','-c:v','libx264','-crf','18',
             '-pix_fmt','yuv420p','-movflags','+faststart',str(path)]
    log_path=path.with_suffix(path.suffix+'.ffmpeg.log')
    with log_path.open('xb') as log:
        process=subprocess.Popen(command,stdin=subprocess.PIPE,stdout=subprocess.DEVNULL,stderr=log)
        try:
            in_image=(xy[...,0]>=0)&(xy[...,0]<w)&(xy[...,1]>=0)&(xy[...,1]<h)&visible.astype(bool)
            for frame in range(len(rgb)):
                image=Image.fromarray(rgb[frame]); draw=ImageDraw.Draw(image)
                for point in np.flatnonzero(in_image[frame]):
                    color=colors[point]
                    for previous in range(max(0,frame-trail+1),frame):
                        if in_image[previous:previous+2,point].all():
                            draw.line([tuple(xy[previous,point]),tuple(xy[previous+1,point])],fill=color,width=2)
                    x,y=xy[frame,point]
                    draw.ellipse((x-3,y-3,x+3,y+3),fill=color)
                    if rigid_inliers is not None and rigid_inliers[frame,point]:
                        draw.rectangle((x-6,y-6,x+6,y+6),outline=(255,255,255),width=2)
                    draw.text((x+4,y+3),'%.2f'%confidence[frame,point],fill=color)
                draw.rectangle((0,0,w,min(30,h)),fill=(0,0,0))
                draw.text((4,2),str(label),fill=(255,255,255))
                draw.text((4,15),'frame %d/%d | t=%.3fs'%(frame+1,len(rgb),timestamps[frame]),fill=(255,255,255))
                if rigid_inliers is not None and rigid_inliers[frame].any():
                    draw.text((4,32),'white box: delayed rigid inlier event',fill=(255,255,255))
                process.stdin.write(np.asarray(image).tobytes())
            process.stdin.close()
            if process.wait()!=0:
                raise RuntimeError('Video encoder failed; inspect '+str(log_path))
        except BaseException:
            if process.poll() is None:
                process.terminate()
            process.wait()
            if not process.stdin.closed:
                process.stdin.close()
            raise
    return path


def render_comparison(rgb,timestamps,arrays,path,trail=20):
    """Four panels from recorded causal outputs, never rerun or fabricate geometry.

    Geometry panel shows candidate C3 (delayed, age-decayed projected correction),
    not an independently evaluated Track4World run. Unavailable points stay hidden.
    """
    path=Path(path)
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    candidates=np.asarray(arrays['candidates'])
    valid=np.asarray(arrays['candidate_valid'])
    probability=np.asarray(arrays['candidate_visibility'])
    processed=np.asarray(arrays['processed'])
    xy=np.asarray(arrays['xy'])
    if candidates.shape!=xy.shape[:2]+(4,2) or valid.shape!=xy.shape[:2]+(4,) or probability.shape!=valid.shape:
        raise ValueError('Comparison requires complete four-candidate diagnostics')
    if processed.shape!=xy.shape[:2] or not np.isin(valid,[0,1]).all() or not np.isin(processed,[0,1]).all():
        raise ValueError('Invalid comparison masks')
    if not np.isfinite(candidates[valid.astype(bool)]).all():
        raise ValueError('Non-finite valid comparison candidate')
    # Missing geometry deliberately uses NaN in saved diagnostics. Only rendering
    # gets finite hidden placeholders; masks continue to hide unavailable points.
    candidates=np.where(valid[...,None].astype(bool),candidates,0.)
    threshold=float(arrays['visibility_threshold'])
    if not np.isfinite(threshold) or not 0<=threshold<=1:
        raise ValueError('Invalid comparison visibility threshold')
    panels=[(xy,np.zeros(xy.shape[:2],bool),np.zeros(xy.shape[:2]),'RGB input'),
            (candidates[:,:,0],valid[:,:,0].astype(bool)&processed.astype(bool)&(probability[:,:,0]>=threshold),
             probability[:,:,0],'Raw '+str(arrays['tracker_source'])+' | visibility'),
            (candidates[:,:,2],valid[:,:,2].astype(bool)&processed.astype(bool)&(probability[:,:,2]>=threshold),
             probability[:,:,2],'C3 delayed projected correction | absent = unavailable'),
            (xy,arrays['visible'],arrays['confidence'],'GeoVerify fused | confidence')]
    # Stream each panel to disk: avoid allocating a second full-resolution video.
    with tempfile.TemporaryDirectory(prefix='comparison-',dir=str(path.parent)) as folder:
        paths=[]
        for i,(points,visible,confidence,label) in enumerate(panels):
            panel=Path(folder)/('%d.mp4'%i)
            render_overlay(rgb,timestamps,points,visible,confidence,panel,label,trail,
                           rigid_inliers=rigid_event_markers(arrays) if i==3 else None)
            paths.append(panel)
        command=['ffmpeg','-hide_banner','-loglevel','error','-n']
        for panel in paths:
            command+=['-i',str(panel)]
        command+=['-filter_complex','[0:v][1:v]hstack[top];[2:v][3:v]hstack[bottom];[top][bottom]vstack[out]',
                  '-map','[out]','-an','-c:v','libx264','-crf','18','-pix_fmt','yuv420p',
                  '-movflags','+faststart',str(path)]
        log_path=path.with_suffix(path.suffix+'.ffmpeg.log')
        with log_path.open('xb') as log:
            result=subprocess.run(command,stdout=subprocess.DEVNULL,stderr=log)
        if result.returncode:
            raise RuntimeError('Comparison encoder failed; inspect '+str(log_path))
    return path
