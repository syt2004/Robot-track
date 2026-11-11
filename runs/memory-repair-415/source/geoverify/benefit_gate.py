"""Predict the marginal spatial benefit of current geometric corrections.

Only causal inference evidence enters the predictor. Labels are constructed
outside this module from TRAIN coordinates. C1/C4 form the 2D counterfactual;
C2/C3 are the geometric intervention. No changes to visibility or raw history.
"""
import torch
from torch import nn
from .calibration import canonical_features

class BenefitGate(nn.Module):
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(64,32),nn.SiLU(),nn.Linear(32,1))
        nn.init.zeros_(self.net[-1].weight);nn.init.zeros_(self.net[-1].bias)
        self.register_buffer('mean',torch.zeros(64));self.register_buffer('std',torch.ones(64))

    @staticmethod
    def inputs(features,visual_logits,valid,weights):
        valid=valid.bool();f=canonical_features(features,valid)
        f=torch.where(valid[...,None],f,torch.zeros_like(f))
        relative=torch.where(valid,visual_logits-visual_logits[...,0,None],0).clamp(-30,30)
        return torch.cat([f.flatten(-2),relative,valid.float(),weights],dim=-1)

    def forward(self,x):
        return self.net(((x-self.mean)/self.std.clamp_min(.01)).clamp(-12,12)).squeeze(-1)

    @staticmethod
    def reference(weights,candidates,valid):
        nong=weights.clone();nong[...,1:3]=0
        nong=nong/nong.sum(-1,keepdim=True).clamp_min(1e-12)
        xy=(nong[...,None]*torch.where(valid[...,None],candidates,0)).sum(-2)
        return nong,xy

    def apply(self,output,features,visual_logits,candidates,valid):
        x=self.inputs(features,visual_logits,valid,output['weights'])
        gain=self(x)
        # Identity at initialization; predicted harm attenuates the correction.
        gate=torch.exp(gain.clamp(min=-20,max=0))
        available=valid[...,1:3].any(-1)
        gate=torch.where(available,gate,torch.ones_like(gate))
        nong,_=self.reference(output['weights'],candidates,valid)
        w=gate[...,None]*output['weights']+(1-gate[...,None])*nong
        result=dict(output)
        result.update(weights=w,xy=(w[...,None]*torch.where(valid[...,None],candidates,0)).sum(-2),
                      confidence=(w*output['reliability_logits'].sigmoid()).sum(-1),
                      predicted_geometry_gain=gain,geometry_gate=gate)
        return result
