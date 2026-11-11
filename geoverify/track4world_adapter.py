"""Run only inside the isolated Track4World worker environment.

Select the physical GPU with CUDA_VISIBLE_DEVICES before launching the worker;
the official infer_pair hardcodes cuda:0 in several places.
"""
import json
from pathlib import Path
import sys
import numpy as np
from geoverify.track4world_geometry import extract_pairwise


class Track4WorldAdapter:
    def __init__(self, root, checkpoint, metric_scale=True,backbone_directory=None):
        import torch
        root = Path(root).resolve()
        checkpoint = Path(checkpoint).resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        config = json.loads((root/'track4world/config/eval/v1.json').read_text())
        if not torch.cuda.is_available():
            raise RuntimeError('Official Track4World infer_pair requires CUDA')
        torch.cuda.set_device(0)
        sys.path.insert(0,str(root))
        from track4world.nets import model as model_module
        from contextlib import nullcontext
        from geoverify.local_backbone import local_hub_model,checkpoint_convnext_initializer
        context=(nullcontext() if backbone_directory is None else
                 local_hub_model(model_module,'DepthAnything3',backbone_directory,
                                 'depth-anything/DA3NESTED-GIANT-LARGE-1.1'))
        with context,checkpoint_convnext_initializer(model_module):
            self.model = model_module.Track4World(**config['model'],seqlen=16,use_3d=True,
                                                 use_model='depthanythingv3')
        state = torch.load(str(checkpoint),map_location='cpu',weights_only=True)
        missing, unexpected = self.model.load_pretrained_with_remap(state)
        # Only the independently pretrained nested metric branch may be absent.
        disallowed = [k for k in missing if not k.startswith('backbone.model.da3_metric.')]
        if unexpected or disallowed:
            raise RuntimeError('Track4World checkpoint incompatible: missing=%s unexpected=%s' %
                               (disallowed,unexpected))
        if backbone_directory is not None and (Path(backbone_directory)/'metric_source.json').is_file():
            metric_keys={k for k in self.model.state_dict() if k.startswith('backbone.model.da3_metric.')}
            if set(missing)!=metric_keys:
                raise RuntimeError('Metric-only loading requires every other parameter from Track4World and metric branch unchanged')
        self.model = self.model.cuda().eval()
        self.model.requires_grad_(False)
        self.model.use_metric_scale = bool(metric_scale)
        self.torch = torch

    def run_window(self, rgb, query_xy):
        """Already-received RGB uint8 [T,H,W,3], query coordinates at frame 0.

        Returns sparse trajectories in this window's gauge, never a persistent
        global frame. Worker scheduling/availability timestamps belong to caller.
        """
        rgb = np.asarray(rgb)
        if rgb.ndim != 4 or rgb.shape[-1] != 3 or rgb.dtype != np.uint8:
            raise ValueError('Expected uint8 RGB [T,H,W,3]')
        if not 2 <= len(rgb) <= 16:
            raise ValueError('History window must contain 2 to 16 frames')
        torch = self.torch
        video = torch.as_tensor(rgb.copy(),device='cuda:0').permute(0,3,1,2)[None].float()
        from track4world.nets import blocks
        from geoverify.bounded_attention import bounded_block_attention
        from geoverify.geometry_intrinsics import capture_da3_intrinsics,resize_unpad_intrinsics
        with capture_da3_intrinsics(self.model.backbone) as cameras, bounded_block_attention(blocks) as attention_stats, torch.inference_mode(), torch.autocast(device_type='cuda',dtype=torch.float16):
            geometry,motion = self.model.infer_pair(
                video,iters=4,tracking3d=True,is_training=False,
                force_projection=False,aligned_scene_flow=False,apply_mask=False)
        def cpu(value):
            return value.detach().float().cpu().numpy()
        g = {k:cpu(geometry[k]) for k in ('points','mask','intrinsics','camera_poses')}
        if len(cameras)!=1 or cameras[0][0].shape[1]!=len(rgb):
            raise RuntimeError('Expected one DA3 calibration per window frame')
        original_intrinsics=g['intrinsics'].copy()
        padder=blocks.InputPadder(video.shape)
        g['intrinsics']=resize_unpad_intrinsics(cameras[0][0],cameras[0][1],rgb.shape[1:3],padder._pad)
        m = {k:cpu(motion[k]) for k in ('flow_2d','flow_3d','visconf_maps_e')}
        result = extract_pairwise(g,m,query_xy,force_projection=False,aligned_scene_flow=False)
        result['intrinsics_original_normalized']=original_intrinsics
        result['intrinsics_da3_depth_grid']=cameras[0][0]
        result['intrinsics_da3_depth_hw']=np.asarray(cameras[0][1])
        result['intrinsics_padding']=np.asarray(padder._pad)
        result['intrinsics_protocol']=np.asarray('da3_per_frame_resize_half_pixel_unpad_v1')
        result['attention_chunked_calls'] = attention_stats['chunked_calls']
        result['attention_max_batch'] = attention_stats['max_batch']
        result['metric_scale_enabled'] = self.model.use_metric_scale
        return result
