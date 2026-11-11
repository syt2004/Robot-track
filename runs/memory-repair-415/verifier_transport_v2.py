"""Reuse uint8 and float RGB buffers; rank_window copied from the frozen adapter.
Only video preparation changes. GPU float values, layout, model and ranking math match.
"""
import numpy as np
import torch
from geoverify.adapters import masked_candidate_attention
class RGBFloatTransport:
 def __init__(self,device):self.device=device;self.host=None;self.done=None
 def prepare(self,rgb):
  assert rgb.dtype==np.uint8 and rgb.ndim==4 and rgb.shape[-1]==3
  if self.done is not None:self.done.synchronize()
  capacity=(max(24,len(rgb)),*rgb.shape[1:])
  if self.host is None or tuple(self.host.shape)!=capacity:
   self.host=torch.empty(capacity,dtype=torch.uint8,pin_memory=True)
   self.gpu=torch.empty(capacity,dtype=torch.uint8,device=self.device)
   self.floating=torch.empty(capacity,dtype=torch.float32,device=self.device)
   self.done=torch.cuda.Event()
  self.host[:len(rgb)].copy_(torch.from_numpy(rgb))
  self.gpu[:len(rgb)].copy_(self.host[:len(rgb)],non_blocking=True)
  self.floating[:len(rgb)].copy_(self.gpu[:len(rgb)],non_blocking=True)
  return self.floating[:len(rgb)].permute(0,3,1,2)[None]
 def finish(self):self.done.record()

def prepared_rank_window(self, rgb_window, queries_txy, candidates, valid=None):
    """Only rank the last frame of a received history window.

    Queries are reanchored to frame 0 of this window. candidates: [T,N,M,2].
    Never publish revised historical predictions as online results.
    """
    torch = self.torch
    if len(rgb_window) != len(candidates):
        raise ValueError('Video/candidate temporal length mismatch')
    candidates=np.asarray(candidates)
    if candidates.ndim!=4 or candidates.shape[-1]!=2:
        raise ValueError('Candidates must have shape [T,N,M,2]')
    if valid is None:
        valid=np.isfinite(candidates).all(-1)
    valid=np.asarray(valid,bool)
    if valid.shape!=candidates.shape[:-1] or not valid.any(-1).all():
        raise ValueError('Invalid candidate availability')
    if not np.isfinite(candidates[valid]).all():
        raise ValueError('Non-finite valid verifier candidate')
    # Sampling invalid coordinates is unsafe; copy the first valid endpoint
    # but mask those descriptors in every spatial attention layer.
    first=valid.argmax(-1)
    safe=candidates.copy()
    fallback=np.take_along_axis(safe,first[...,None,None],axis=2)
    safe=np.where(valid[...,None],safe,fallback)
    if not hasattr(self, '_rgb_float_transport'): self._rgb_float_transport = RGBFloatTransport(self.device)
    video = self._rgb_float_transport.prepare(rgb_window)
    tracks = torch.as_tensor(safe,device=self.device,dtype=torch.float32)[None]
    available=torch.as_tensor(valid,device=self.device)[None]
    queries = torch.as_tensor(queries_txy,device=self.device,dtype=torch.float32)[None]
    if not bool((queries[...,0] == 0).all()):
        raise ValueError('Verifier queries must reference the local window anchor')
    with torch.inference_mode(), masked_candidate_attention(self.model,available):
        ranks = self.model.efficient_forward(video,tracks,queries)
    result = ranks[0,-1].masked_fill(~available[0,-1],-torch.inf).float().cpu().numpy()
    self._rgb_float_transport.finish()
    return result


def install():
 from geoverify.adapters import VerifierAdapter
 original=VerifierAdapter.rank_window
 VerifierAdapter.rank_window=prepared_rank_window
 return original
