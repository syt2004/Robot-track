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
