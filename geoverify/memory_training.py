"""Differentiable equivalent of Track-On2 track_frame, with frozen weights.

The caller owns model/source initialization. No labels enter this function.
Returns correlation logits as well as predictions for trajectory supervision.
"""
import torch


def track_frame_with_grad(model, q_init, temporal_mask, point_memory,
                          frame_features, height, width):
    from utils.coord_utils import indices_to_coords

    f4, f8, f16, f32, fused = frame_features
    q = q_init.unsqueeze(0).clone()
    memory = point_memory.clone()
    mask = torch.cat((temporal_mask, torch.zeros(
        len(q_init), 1, device=q.device, dtype=torch.bool)), dim=1)
    for i in range(model.decoder_layer_num):
        q = model.feature_attention[i](q, fused, fused)
        q = model.query_attention[i](q, q, q)
        qkv = torch.cat((memory, q[0, :, None]), dim=1)
        qkv = model.memory_attention[i](qkv + model.t_embedding,
                                        qkv + model.t_embedding, qkv, mask)
        q = qkv[:, -1][None]
        memory = qkv[:, :-1]
    q = model.projection1(q)
    c1 = model.multiscale_correlation(q, f4, f8, f16, f32)
    q, _, _, _ = model.reranking_head(q, f4, f8, f16, f32, c1)
    q = model.projection2(q)
    c2 = model.multiscale_correlation(q, f4, f8, f16, f32)
    patch = indices_to_coords(c2.argmax(-1).unsqueeze(1),
                              model.input_size, model.stride).squeeze(1)
    offsets, visibility, uncertainty = model.prediction_head(q, f4, f8, f16, f32, patch)
    scale = q.new_tensor([width / model.W, height / model.H])
    return {'xy': (patch[0] + offsets[-1]) * scale,
            'visibility_logit': visibility, 'q_new': q[0],
            'correlation_logits': c2[0], 'uncertainty_logit': uncertainty}
