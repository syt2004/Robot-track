"""Preserve per-frame DA3 rays through Track4World resize and symmetric padding."""
from contextlib import contextmanager
import numpy as np


def resize_unpad_intrinsics(intrinsics, depth_hw, output_hw, padding):
    """Pixel-center convention of process_geometry and align_corners=False.

    padding is the actual upstream InputPadder [left,right,top,bottom].
    Returns normalized intrinsics expected by extract_pairwise; no XYZ fitting.
    """
    k=np.asarray(intrinsics,dtype=np.float64).copy()
    dh,dw=depth_hw;h,w=output_hw;left,right,top,bottom=padding
    if (k.ndim!=4 or k.shape[0]!=1 or k.shape[-2:]!=(3,3)
            or not np.isfinite(k).all() or min(dh,dw,h,w)<=0
            or min(padding)<0 or not np.allclose(k[...,2,:],[0,0,1])
            or (k[...,0,0]<=0).any() or (k[...,1,1]<=0).any()):
        raise ValueError('Invalid DA3 camera intrinsics or image dimensions')
    sx=(w+left+right)/dw;sy=(h+top+bottom)/dh
    k[...,0,:]*=sx;k[...,1,:]*=sy
    k[...,0,2]+=(sx-1)/2-left
    k[...,1,2]+=(sy-1)/2-top
    k[...,0,:]/=w;k[...,1,:]/=h
    return k


@contextmanager
def capture_da3_intrinsics(backbone):
    """Capture actual depth-grid calibration without changing model predictions."""
    original=backbone.inference_v2
    owned='inference_v2' in backbone.__dict__
    saved=backbone.__dict__.get('inference_v2')
    records=[]
    def wrapped(*args,**kwargs):
        result=original(*args,**kwargs)
        records.append((result['intrinsics'].detach().float().cpu().numpy().copy(),
                        tuple(result['depth'].shape[-2:])))
        return result
    backbone.inference_v2=wrapped
    try:yield records
    finally:
        if owned:backbone.inference_v2=saved
        else:delattr(backbone,'inference_v2')
