"""Use upstream DPT/DualDPT's existing frame-chunk API; no weight/source edits."""
from contextlib import contextmanager
@contextmanager
def depth_head_chunks(model,chunk_size=1):
 if not isinstance(chunk_size,int) or chunk_size<1:raise ValueError('Positive integer chunk size required')
 patched=[];stats={'chunk_size':chunk_size,'heads':[],'calls':[]}
 try:
  for name,module in model.named_modules():
   cls=type(module)
   if cls.__name__ not in ('DPT','DualDPT') or not cls.__module__.startswith('depth_anything_3.model.'):continue
   if module.training:raise ValueError('Inference-only depth head chunking')
   original=module.forward
   def wrap(feats,H,W,patch_start_idx,chunk_size=8,_original=original,_name=name,**kwargs):
    shape=list(feats[0][0].shape)
    if shape[0]!=1:raise ValueError('Validated B=1 geometry windows only')
    stats['calls'].append(dict(head=_name,features_shape=shape,hw=[int(H),int(W)]))
    return _original(feats,H,W,patch_start_idx,chunk_size=stats['chunk_size'],**kwargs)
   patched.append((module,original));module.forward=wrap;stats['heads'].append(dict(name=name,type=cls.__name__,module=cls.__module__))
  if not patched:raise ValueError('No upstream DPT depth heads found')
  yield stats
 finally:
  for module,original in reversed(patched):module.forward=original
