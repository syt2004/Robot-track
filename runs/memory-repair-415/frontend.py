"""Bound unused CUDA allocator cache; preserve all frontend model equations."""
import torch
from geoverify.run_dataset import main
original=torch.cuda.empty_cache
def bounded_cache():
    if torch.cuda.memory_reserved()>=4*2**30 and torch.cuda.memory_reserved()-torch.cuda.memory_allocated()>=2**30:
        original()
if __name__=='__main__':
    torch.cuda.set_per_process_memory_fraction(6*2**30/torch.cuda.get_device_properties(0).total_memory)
    torch.cuda.empty_cache=bounded_cache
    try:main()
    finally:torch.cuda.empty_cache=original
