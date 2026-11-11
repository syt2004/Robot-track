"""Point-frame exposure independent of query cohort and cache shard size."""
import numpy as np

def labeled_rows(arrays):
    mask=np.asarray(arrays['target_valid'],bool)
    return {k:np.asarray(v)[mask] for k,v in arrays.items()}

def epoch_indices(count,batch_points,rng):
    if count<1 or batch_points<1:raise ValueError('Positive sample and batch sizes required')
    order=rng.permutation(count)
    for start in range(0,count,batch_points):yield order[start:start+batch_points]
