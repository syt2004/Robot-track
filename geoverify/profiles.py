"""Explicit tracker variants. Never silently fall back between checkpoints."""
from types import SimpleNamespace


def tracker_args(variant='trackon_r_dinov3'):
    # DINOv2 settings match upstream track-on2/config/test_dinov2.yaml.
    variants = {
        'trackon_r_dinov3': ('dinov3_s_plus', 1.143),
        'trackon2_dinov2': ('dinov2_s', 1.0),
    }
    if variant not in variants:
        raise ValueError('Unknown tracker variant: ' + str(variant))
    backbone, scale = variants[variant]
    return SimpleNamespace(input_size=[384, 512], M=24, D=256, K=16,
                           decoder_layer_num=3, predicton_head_layer_num=3,
                           rerank_layer_num=3, vit_backbone=backbone,
                           vit_upsample_factor=scale, grad_checkpoint=True,
                           M_i=72, delta_v=0.8)


def validate_checkpoint_backbone(state, variant):
    """Catch mismatched variants before constructing/downloading a backbone.

    A missing frozen encoder is allowed by the upstream loader;
    all other missing/unexpected keys must still be checked during loading.
    """
    backbone = tracker_args(variant).vit_backbone.split('_')[0]
    other = 'dinov2' if backbone == 'dinov3' else 'dinov3'
    if any('.' + other + '.' in key for key in state):
        raise ValueError('Checkpoint backbone does not match ' + variant)
