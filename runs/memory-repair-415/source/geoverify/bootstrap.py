"""Training-data rollout policy, never a substitute for a trained inference head."""
import numpy as np


class VerifierBootstrapPolicy:
    output_kind='training_bootstrap'

    def predict(self,features,visual_logits,packet):
        valid=np.asarray(packet['valid'],bool)
        scores=np.asarray(visual_logits,float)
        xy=np.asarray(packet['candidates'],float)
        visibility=np.asarray(packet['candidate_visibility'],float)
        if scores.shape!=valid.shape or xy.shape!=valid.shape+(2,) or visibility.shape!=valid.shape:
            raise ValueError('Bootstrap candidate shape mismatch')
        if not valid.any(-1).all() or not np.isfinite(scores[valid]).all() or not np.isfinite(xy[valid]).all():
            raise ValueError('Bootstrap requires finite available candidates')
        if not np.isfinite(visibility[valid]).all() or ((visibility[valid]<0)|(visibility[valid]>1)).any():
            raise ValueError('Invalid bootstrap visibility')
        logits=np.where(valid,scores,-np.inf)
        weights=np.exp(logits-logits.max(-1,keepdims=True))
        weights/=weights.sum(-1,keepdims=True)
        # Legacy cache rollout mixes visibility; v2 uses current C1 at inference. The
        # reliability output is explicitly untrained (0.5), not claimed accuracy.
        return dict(xy=(weights[...,None]*np.where(valid[...,None],xy,0)).sum(-2),
                    visibility=(weights*np.where(valid,visibility,0)).sum(-1),
                    confidence=np.full(valid.shape[:-1],.5),weights=weights)
