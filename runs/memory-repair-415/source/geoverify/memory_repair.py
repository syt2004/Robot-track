"""Geometry-conditioned residual in tracker memory space, not output space."""
import torch
from torch import nn
import torch.nn.functional as F

class GeometryMemoryRepair(nn.Module):
    def __init__(self, dimension=256, hidden=128, maximum_relative_change=.25):
        super().__init__()
        self.maximum_relative_change=maximum_relative_change
        self.net=nn.Sequential(nn.LayerNorm(2*dimension+8),nn.Linear(2*dimension+8,hidden),nn.GELU(),nn.Linear(hidden,dimension))
        nn.init.zeros_(self.net[-1].weight);nn.init.zeros_(self.net[-1].bias)
    def forward(self,memory,sampled,evidence,valid):
        safe=torch.where(valid[...,None],sampled,torch.zeros_like(sampled))
        ev=torch.where(valid[...,None],evidence,torch.zeros_like(evidence))
        residual=self.net(torch.cat((memory,safe,ev),-1))
        # A correction is a bounded displacement from the original slot token.
        # This is a state-space constraint, not a gate on final output positions.
        budget=self.maximum_relative_change*memory.norm(dim=-1,keepdim=True).clamp_min(1e-6)
        scale=(budget/residual.norm(dim=-1,keepdim=True).clamp_min(1e-6)).clamp(max=1.)
        residual=residual*scale
        return memory+torch.where(valid[...,None],residual,torch.zeros_like(residual))

def sample_features(model,fused,xy,height,width):
    grid=xy*xy.new_tensor([2/width,2/height])-1
    fmap=fused.view(1,model.Hf,model.Wf,model.D).permute(0,3,1,2)
    return F.grid_sample(fmap,grid[None,:,None],align_corners=False,padding_mode='border')[0,:,:,0].T


def bound_to_reference(proposal, reference, maximum_relative_change=.25):
    """Differentiable projection into a per-token ball around an independent state."""
    delta = proposal - reference
    radius = maximum_relative_change * reference.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    scale = (radius / delta.norm(dim=-1, keepdim=True).clamp_min(1e-6)).clamp(max=1.)
    return reference + delta * scale
