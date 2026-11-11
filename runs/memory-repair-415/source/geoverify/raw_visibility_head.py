"""Trainable candidate correction shared by cache training and online inference."""
import torch
from torch import nn
from torch.nn import functional as F

from geoverify.feature_schema import FEATURE_NAMES


HEAD_ARCHITECTURE = 'raw2d_temporal_setrank1_current_visibility_position1000_excess1000_v1'


def canonical_features(features, valid):
    """Age describes usable geometric evidence, not time since expired evidence.

    Raw cache/input schema remains unchanged. Apply this same transformation
    before fitting normalization statistics and in every model forward call.
    Missing evidence uses -1, consistent with the feature schema. Raw wall-clock
    ages remain in replay diagnostics; expiry and causality checks are unchanged.
    """
    result=features.clone()
    available=valid[...,1:3].bool().any(dim=-1)
    age=features[...,11]
    result[...,11]=torch.where(available[...,None],age,torch.full_like(age,-1.))
    return result


class GeometryCalibrationHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(len(FEATURE_NAMES),128),nn.GELU(),
                                     nn.Linear(128,64),nn.GELU(),nn.Linear(64,3))
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
        if not bool(valid[...,0].all()):
            raise ValueError('Current Track-On candidate C1 must be valid')
        if not bool(valid.any(dim=-1).all()):
            raise ValueError('No valid candidate for one or more queries')
        if not bool(torch.isfinite(features[valid]).all() and torch.isfinite(candidates[valid]).all()
                    and torch.isfinite(visual_logits[valid]).all()):
            raise ValueError('Non-finite data in a valid candidate')
        vis_valid = candidate_visibility[valid]
        if not bool(torch.isfinite(vis_valid).all() and (vis_valid>=0).all() and (vis_valid<=1).all()):
            raise ValueError('Visibility must be probability in [0,1]')
        features = canonical_features(features,valid)
        normalized = (torch.where(valid[...,None],features,0)-self.feature_mean)/self.feature_std.clamp_min(1e-5)
        correction, reliability, visibility_correction = self.network(normalized).unbind(-1)
        temperature = self.log_temperature.exp().clamp(.05,20)
        logits = (torch.where(valid,visual_logits,0)+correction)/temperature
        logits = logits.masked_fill(~valid,-torch.inf)
        weights = logits.softmax(dim=-1)
        xy = (weights[...,None]*torch.where(valid[...,None],candidates,0)).sum(-2)
        # C2/C3 visibility belongs to an older geometry window. Keep it in
        # features as evidence, but anchor the current prediction to C1.
        base_visibility = candidate_visibility[...,0]
        # Separate visibility evidence from spatial candidate correctness.
        # Clamp permits correction even when all source probabilities are 0/1.
        visibility_logits = torch.logit(base_visibility.clamp(1e-6,1-1e-6)) + (weights*visibility_correction).sum(-1)
        visibility = visibility_logits.sigmoid()
        confidence = (weights*reliability.sigmoid()).sum(-1)
        return dict(xy=xy,visibility=visibility,confidence=confidence,weights=weights,
                    logits=logits,reliability_logits=reliability,visibility_logits=visibility_logits)


def acceptable_rank_loss(logits, distance, valid, image_diagonal):
    """Negative log mass of candidates within one isotropic256 pixel of best.

    Square TRAIN images make this exactly one pixel at resolution256. For
    rectangles it is diagonal normalization, not per-axis benchmark distance.
    Invalid candidates cannot enter the target set. No tie preference for C1.
    """
    valid = valid.bool()
    diagonal = torch.as_tensor(image_diagonal, device=distance.device, dtype=distance.dtype)
    radius = diagonal / (256. * 2. ** .5)
    nearest = distance.masked_fill(~valid, torch.inf).min(-1).values
    acceptable = valid & (distance <= (nearest + radius)[..., None])
    all_logits = logits.masked_fill(~valid, -torch.inf)
    set_logits = logits.masked_fill(~acceptable, -torch.inf)
    return (torch.logsumexp(all_logits, dim=-1) - torch.logsumexp(set_logits, dim=-1)).mean()


def calibration_loss(output, candidates, valid, target_xy, target_visible, target_valid,
                     image_diagonal, reliability_radius=4.):
    """Only independently labeled, visible GT supervises positions/ranking.

    Occluded points supervise visibility but do not fabricate hidden positions.
    Pixel threshold for reliability labels is explicit; regression is normalized.
    """
    labeled = target_valid.bool()
    spatial = labeled & target_visible.bool() & torch.isfinite(target_xy).all(-1)
    zero = output['xy'].sum()*0
    rank, regression, reliability, visibility, excess = zero,zero,zero,zero,zero
    if bool(spatial.any()):
        # Boolean masking collapses every leading batch dimension first;
        # insert the candidate axis afterwards (works for both N and T,N).
        distance = torch.linalg.vector_norm(candidates[spatial]-target_xy[spatial].unsqueeze(-2),dim=-1)
        distance = distance.masked_fill(~valid[spatial],torch.inf)
        rank_diagonal = torch.as_tensor(image_diagonal, device=target_xy.device, dtype=target_xy.dtype)
        if rank_diagonal.ndim:
            rank_diagonal = rank_diagonal[spatial]
        rank = acceptable_rank_loss(output['logits'][spatial], distance, valid[spatial], rank_diagonal)
        diagonal = torch.as_tensor(image_diagonal,device=target_xy.device,dtype=target_xy.dtype)
        if diagonal.ndim:
            diagonal = diagonal[spatial].unsqueeze(-1)
        regression = F.smooth_l1_loss(output['xy'][spatial]/diagonal,
                                     target_xy[spatial]/diagonal,beta=.01)
        # Extra cost only when this labeled visible point becomes worse than C1.
        # Same normalized Huber units/beta as regression: harmful positional
        # gradients receive double weight, beneficial ones retain their weight.
        # This is a TRAIN objective, not a guarantee of non-degradation at inference.
        point_loss = F.smooth_l1_loss(output['xy'][spatial]/diagonal,
                                     target_xy[spatial]/diagonal,beta=.01,reduction='none').mean(-1)
        raw_loss = F.smooth_l1_loss(candidates[spatial][:,0]/diagonal,
                                   target_xy[spatial]/diagonal,beta=.01,reduction='none').mean(-1)
        excess = F.relu(point_loss-raw_loss).mean()
        selected = valid[spatial]
        reliability = F.binary_cross_entropy_with_logits(
            output['reliability_logits'][spatial][selected],
            (distance[selected]<=reliability_radius).float())
    if bool(labeled.any()):
        visibility = F.binary_cross_entropy_with_logits(output['visibility_logits'][labeled],
                                                        target_visible[labeled].float())
    total = rank+1000*regression+visibility+reliability+1000*excess
    return total,dict(rank=rank,track=regression,visibility=visibility,reliability=reliability,excess=excess)
