"""Read-only capture of pretrained Verifier channel interactions."""
from contextlib import contextmanager
import numpy as np
import torch
from geoverify.adapters import VerifierAdapter

@contextmanager
def projected_outputs(model):
    captured={};handles=[]
    def hook(name):
        def receive(module,args,output):
            if name in captured:raise RuntimeError('Repeated projection call')
            captured[name]=output.detach().clone()
        return receive
    try:
        for name in ('q_projection','f_projection'):
            handles.append(getattr(model,name).register_forward_hook(hook(name)))
        yield captured
    finally:
        for handle in handles:handle.remove()

def interactions(captured,valid):
    q,f=captured['q_projection'],captured['f_projection']
    if q.shape!=f.shape or q.ndim!=5 or q.shape[0]!=1 or q.shape[3]!=valid.shape[-1]:
        raise ValueError('Verifier projected descriptor layout changed')
    if q.shape[2:4]!=valid.shape[-2:]:raise ValueError('Query/candidate dimensions changed')
    q=torch.nn.functional.normalize(q[:,-1],p=2,dim=-1)
    f=torch.nn.functional.normalize(f[:,-1],p=2,dim=-1)
    descriptor=(q*f)[0]
    mask=torch.as_tensor(valid[-1],device=descriptor.device,dtype=torch.bool)
    return torch.where(mask[...,None],descriptor,0)

class DescriptorVerifier(VerifierAdapter):
    def rank_window(self,rgb_window,queries_txy,candidates,valid=None):
        if valid is None:valid=np.isfinite(candidates).all(-1)
        self.last_descriptor=None;self.last_closure_error=None
        with torch.inference_mode(),projected_outputs(self.model) as captured:
            scores=super().rank_window(rgb_window,queries_txy,candidates,valid)
            descriptor=interactions(captured,np.asarray(valid,bool))
            closure=(descriptor.sum(-1)*self.model.sim_scale()).float().cpu().numpy()
            mask=np.asarray(valid,bool)[-1]
            np.testing.assert_allclose(closure[mask],scores[mask],atol=1e-5,rtol=1e-6)
            value=descriptor.float().cpu().numpy()
            assert np.isfinite(value).all() and not value[~mask].any()
            self.last_descriptor=value
            self.last_closure_error=float(np.max(np.abs(closure[mask]-scores[mask])))
        return scores
