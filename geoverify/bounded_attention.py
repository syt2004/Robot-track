"""Bound independent SDPA batches without splitting heads or temporal sequences.

Scoped to Track4World blocks; PyTorch 2.5 CUDA SDPA can fail at batch > 65535.
No parameters, normalization, dropout, or attention graph are changed.
"""
from contextlib import contextmanager


@contextmanager
def bounded_block_attention(blocks, limit=32768):
    if not isinstance(limit,int) or not 1<=limit<=65535:
        raise ValueError('Attention batch limit must be within CUDA grid bound')
    original=blocks.F
    stats=dict(chunked_calls=0,max_batch=0)
    class FunctionalProxy:
        def __getattr__(self,name):
            return getattr(original,name)

        def scaled_dot_product_attention(self,q,k,v,*args,**kwargs):
            # The vendored blocks use only q/k/v; never reinterpret masked or
            # broadcast attention calls that this compatibility path cannot audit.
            if q.ndim!=4 or q.shape[0]<=limit or args or kwargs:
                return original.scaled_dot_product_attention(q,k,v,*args,**kwargs)
            if k.shape[0]!=q.shape[0] or v.shape[0]!=q.shape[0]:
                raise ValueError('Chunked attention requires equal independent batch dimensions')
            import torch
            stats['chunked_calls']+=1
            stats['max_batch']=max(stats['max_batch'],q.shape[0])
            return torch.cat([original.scaled_dot_product_attention(q[i:i+limit],k[i:i+limit],v[i:i+limit])
                              for i in range(0,q.shape[0],limit)],dim=0)
    blocks.F=FunctionalProxy()
    try:
        yield stats
    finally:
        blocks.F=original
