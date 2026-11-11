"""Strict probability boundary with a four-ULP float32 rounding allowance.

Only called on output probabilities. Does not change candidate weights, positions,
logits, losses or in-range outputs. Material/non-finite violations still fail.
"""
import numpy as np

def bounded_probability(values,name):
    values=np.asarray(values)
    if values.dtype!=np.float32:
        raise ValueError('Expected float32 probability: '+name)
    tolerance=4*np.finfo(np.float32).eps
    if not np.isfinite(values).all() or (values < -tolerance).any() or (values > 1+tolerance).any():
        raise ValueError('Material probability violation: '+name+' min='+str(values.min())+' max='+str(values.max()))
    return np.clip(values,0,1)
