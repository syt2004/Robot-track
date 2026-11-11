"""Track-On official-code adapters; no approximate tracker implementations.

Run this module in the Track-On environment only. Track4World belongs to its
own process because both upstream repositories define top-level packages.
"""
from pathlib import Path
from types import SimpleNamespace
import sys
from contextlib import contextmanager
import numpy as np
from geoverify.profiles import tracker_args, validate_checkpoint_backbone


@contextmanager
def masked_candidate_attention(model, valid):
    """Mask unavailable candidates in official Verifier spatial attention.

    Keeps all pretrained parameters and temporal logic. Finite placeholders are
    only used for feature sampling; they must not become extra attention votes.
    valid is [B,T,N,M], matching the upstream B*T*N spatial flatten order.
    """
    padding = ~valid.reshape(-1,valid.shape[-1]).bool()
    if bool(padding.all(-1).any()):
        raise ValueError('Verifier needs at least one available candidate per row')
    def mask_inputs(module,args,kwargs):
        if args[1].shape[:2] != padding.shape:
            raise ValueError('Upstream Verifier candidate layout changed')
        kwargs=dict(kwargs)
        kwargs['key_padding_mask']=padding
        return args,kwargs
    handles=[]
    try:
        for layer in model.layers:
            handles.append(layer.mha1.register_forward_pre_hook(mask_inputs,with_kwargs=True))
        yield
    finally:
        for handle in handles:
            handle.remove()


def add_trackon_source(root):
    root = Path(root).resolve()
    if not (root / 'model/trackon_predictor.py').is_file():
        raise FileNotFoundError('Incomplete Track-On checkout: ' + str(root))
    sys.path.insert(0, str(root))


class TrackOnAdapter:
    def __init__(self, root, checkpoint, device='cuda:0', variant='trackon_r_dinov3',backbone_directory=None):
        import torch
        args = tracker_args(variant)
        checkpoint = Path(checkpoint).resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        raw = torch.load(str(checkpoint), map_location='cpu', weights_only=True)
        state = raw
        for key in ('state_dict', 'model', 'model_state', 'ema_state_dict'):
            if isinstance(raw, dict) and isinstance(raw.get(key), dict):
                state = raw[key]
                break
        validate_checkpoint_backbone(state, variant)
        del raw, state
        add_trackon_source(root)
        from model.trackon_predictor import Predictor
        self.torch = torch
        self.device = device
        self.variant = variant
        from contextlib import nullcontext
        context=nullcontext()
        if backbone_directory is not None:
            import importlib
            from geoverify.local_backbone import local_auto_model
            family='dinov2' if variant=='trackon2_dinov2' else 'dinov3'
            module=importlib.import_module('model.vit_adapter.'+family+'_adapter.'+family+'_vit_adapter')
            repository=('facebook/dinov2-small' if family=='dinov2'
                        else 'facebook/dinov3-vits16plus-pretrain-lvd1689m')
            context=local_auto_model(module,backbone_directory,repository)
        with context:
            self.predictor = Predictor(model_args=args, checkpoint_path=str(checkpoint), support_grid_size=0)
        self.predictor = self.predictor.to(device).eval()
        self.predictor.requires_grad_(False)

    def reset(self):
        self.predictor.reset()

    def step(self, rgb, new_queries=None):
        """RGB uint8 [H,W,3]; queries in original image (x,y) pixels."""
        torch = self.torch
        if rgb.ndim != 3 or rgb.shape[-1] != 3 or str(rgb.dtype) != 'uint8':
            raise ValueError('Expected RGB uint8 HWC frame')
        frame = torch.as_tensor(rgb.copy(),device=self.device).permute(2,0,1)[None].float()
        queries = None if new_queries is None else torch.as_tensor(new_queries,device=self.device,dtype=torch.float32)
        # Capture logits at the upstream method boundary without editing memory
        # updates or maintaining a fork of forward_frame. Restore even on errors.
        original = self.predictor.model.track_frame
        captured = {}
        def capture(*args, **kwargs):
            result = original(*args, **kwargs)
            captured['visibility'] = result[1].sigmoid().detach()
            return result
        self.predictor.model.track_frame = capture
        try:
            with torch.inference_mode():
                xy, visible = self.predictor.forward_frame(frame,queries)
        finally:
            self.predictor.model.track_frame = original
        probability = captured.get('visibility', visible.float())
        return {'xy':xy.cpu().numpy(), 'visibility':probability.cpu().numpy(),
                'source': self.variant}


class VerifierAdapter:
    def __init__(self, root, checkpoint, device='cuda:0'):
        import torch
        checkpoint = Path(checkpoint).resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        add_trackon_source(root)
        from verifier import verifier as verifier_module
        from verifier.verifier_predictor import _extract_state_dict, _strip_module_prefix
        args = SimpleNamespace(verifier_input_size=[384,512],verifier_embedding_dim=256,
                               verifier_num_layers=6,local_encoder_num_layers=3,
                               query_critic='feature',T=24)
        state = _strip_module_prefix(_extract_state_dict(torch.load(str(checkpoint),map_location='cpu',weights_only=True)))
        from geoverify.local_backbone import local_verifier_encoder
        with local_verifier_encoder(verifier_module,state):
            self.model = verifier_module.Verifier(args).to(device).eval()
        self.model.load_state_dict(state,strict=True)
        self.model.requires_grad_(False)
        self.torch, self.device = torch, device

    def rank_window(self, rgb_window, queries_txy, candidates, valid=None):
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
        video = torch.as_tensor(rgb_window.copy(),device=self.device).permute(0,3,1,2)[None].float()
        tracks = torch.as_tensor(safe,device=self.device,dtype=torch.float32)[None]
        available=torch.as_tensor(valid,device=self.device)[None]
        queries = torch.as_tensor(queries_txy,device=self.device,dtype=torch.float32)[None]
        if not bool((queries[...,0] == 0).all()):
            raise ValueError('Verifier queries must reference the local window anchor')
        with torch.inference_mode(), masked_candidate_attention(self.model,available):
            ranks = self.model.efficient_forward(video,tracks,queries)
        return ranks[0,-1].masked_fill(~available[0,-1],-torch.inf).float().cpu().numpy()
