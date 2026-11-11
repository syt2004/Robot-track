"""Keep evidence generation on the unmodified stream; expose repaired C1 explicitly.
The frozen base pipeline/history implementation is retained. Its raw input is the
independent shadow stream; only the live C1 proposal is replaced before Verifier.
"""
import numpy as np
from .pipeline import CausalGeoVerifyTracker

class ShadowEvidenceStream:
    def __init__(self,repairing):self.repairing=repairing;self.last_repair=None
    def reset(self):self.last_repair=None;self.repairing.reset()
    def step(self,rgb,new_queries=None):
        repaired=self.repairing.step(rgb,new_queries);self.last_repair=repaired
        return dict(xy=self.repairing.shadow_xy.copy(),visibility=self.repairing.shadow_visibility.copy(),source=repaired['source'])
    def __getattr__(self,name):return getattr(self.repairing,name)

class RepairProposalBuilder:
    def __init__(self,base,stream):self.base=base;self.stream=stream
    def __getattr__(self,name):return getattr(self.base,name)
    def build(self,now):
        # All history, geometric deltas and temporal proposals remain shadow-based.
        packet=self.base.build(now);proposal=self.stream.last_repair
        xy=np.asarray(proposal['xy']);visibility=np.asarray(proposal['visibility'])
        if xy.shape!=packet['candidates'][:,0].shape or visibility.shape!=packet['candidate_visibility'][:,0].shape:
            raise ValueError('Repaired query identity/shape mismatch')
        if not np.isfinite(xy).all() or not np.isfinite(visibility).all() or ((visibility<0)|(visibility>1)).any():
            raise ValueError('Invalid repaired proposal')
        packet['candidates'][:,0]=xy;packet['candidate_visibility'][:,0]=visibility
        return packet

class ShadowHistoryPipeline(CausalGeoVerifyTracker):
    def _step(self,*args,**kwargs):
        if self.evidence_history_policy!='raw_trackon':raise ValueError('Independent shadow evidence requires raw-history policy')
        if not isinstance(self.builder,RepairProposalBuilder):self.builder=RepairProposalBuilder(self.builder,self.trackon)
        output=super()._step(*args,**kwargs)
        output['shadow_xy']=self.trackon.shadow_xy.copy();output['shadow_visibility']=self.trackon.shadow_visibility.copy()
        output['memory_reference_protocol']='independent_shadow_history_v1'
        if self.clock is not None:
            observed=float(self.clock())
            if observed<output['available_timestamp']:raise ValueError('Publication clock moved backwards')
            output['available_timestamp']=observed;self.last_available=observed
        return output
