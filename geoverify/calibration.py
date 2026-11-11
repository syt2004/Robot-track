"""Trainable candidate correction shared by cache training and online inference."""
import torch
from torch import nn
from torch.nn import functional as F

from geoverify.feature_schema import FEATURE_NAMES


class GeometryCalibrationHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(len(FEATURE_NAMES),128),nn.GELU(),
                                     nn.Linear(128,64),nn.GELU(),nn.Linear(64,2))
        # Identity residual at initialization preserves the original ranking.
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)
        self.log_temperature = nn.Parameter(torch.zeros(()))
        self.register_buffer('feature_mean',torch.zeros(len(FEATURE_NAMES)))
        self.register_buffer('feature_std',torch.ones(len(FEATURE_NAMES)))

    def forward(self, features, visual_logits, candidates, candidate_visibility, valid):
        """Leading dimensions arbitrary; last dimensions [M,F], [M], [M,2].

        Invalid candidates may contain NaNs. Each row must contain at least one
        valid candidate. Visibility and spatial correctness remain distinct.
        """
        valid = valid.bool()
        if features.shape[-1] != len(FEATURE_NAMES):
            raise ValueError('Feature schema mismatch')
        if features.shape[:-1] != visual_logits.shape or candidates.shape != (*valid.shape,2):
            raise ValueError('Candidate dimensions mismatch')
        if candidate_visibility.shape != valid.shape or visual_logits.shape != valid.shape:
            raise ValueError('Visibility/logit dimensions mismatch')
        if not bool(valid.any(dim=-1).all()):
            raise ValueError('No valid candidate for one or more queries')
        if not bool(torch.isfinite(features[valid]).all() and torch.isfinite(candidates[valid]).all()
                    and torch.isfinite(visual_logits[valid]).all()):
            raise ValueError('Non-finite data in a valid candidate')
        vis_valid = candidate_visibility[valid]
        if not bool(torch.isfinite(vis_valid).all() and (vis_valid>=0).all() and (vis_valid<=1).all()):
            raise ValueError('Visibility must be probability in [0,1]')
        normalized = (torch.where(valid[...,None],features,0)-self.feature_mean)/self.feature_std.clamp_min(1e-5)
        correction, reliability = self.network(normalized).unbind(-1)
        temperature = self.log_temperature.exp().clamp(.05,20)
        logits = (torch.where(valid,visual_logits,0)+correction)/temperature
        logits = logits.masked_fill(~valid,-torch.inf)
        weights = logits.softmax(dim=-1)
        xy = (weights[...,None]*torch.where(valid[...,None],candidates,0)).sum(-2)
        visibility = (weights*torch.where(valid,candidate_visibility,0)).sum(-1)
        confidence = (weights*reliability.sigmoid()).sum(-1)
        return dict(xy=xy,visibility=visibility,confidence=confidence,weights=weights,
                    logits=logits,reliability_logits=reliability)


def calibration_loss(output, candidates, valid, target_xy, target_visible, target_valid,
                     image_diagonal, reliability_radius=4.):
    """Only independently labeled, visible GT supervises positions/ranking.

    Occluded points supervise visibility but do not fabricate hidden positions.
    Pixel threshold for reliability labels is explicit; regression is normalized.
    """
    labeled = target_valid.bool()
    spatial = labeled & target_visible.bool() & torch.isfinite(target_xy).all(-1)
    zero = output['xy'].sum()*0
    rank, regression, reliability, visibility = zero,zero,zero,zero
    if bool(spatial.any()):
        # Boolean masking collapses every leading batch dimension first;
        # insert the candidate axis afterwards (works for both N and T,N).
        distance = torch.linalg.vector_norm(candidates[spatial]-target_xy[spatial].unsqueeze(-2),dim=-1)
        distance = distance.masked_fill(~valid[spatial],torch.inf)
        best = distance.argmin(-1)
        rank = F.cross_entropy(output['logits'][spatial],best)
        regression = F.smooth_l1_loss(output['xy'][spatial]/image_diagonal,
                                     target_xy[spatial]/image_diagonal,beta=.01)
        selected = valid[spatial]
        reliability = F.binary_cross_entropy_with_logits(
            output['reliability_logits'][spatial][selected],
            (distance[selected]<=reliability_radius).float())
    if bool(labeled.any()):
        visibility = F.binary_cross_entropy(output['visibility'][labeled].clamp(1e-6,1-1e-6),
                                            target_visible[labeled].float())
    total = rank+10*regression+visibility+reliability
    return total,dict(rank=rank,track=regression,visibility=visibility,reliability=reliability)
