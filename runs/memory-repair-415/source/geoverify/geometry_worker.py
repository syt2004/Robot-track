"""Persistent isolated Track4World worker with explicit measured memory profiles."""
import argparse,os,json,time,traceback
from pathlib import Path
import numpy as np
from geoverify.resource_profiles import validate_profile_window

def atomic_json(path,value):
    temp=path.with_suffix('.pending');temp.write_text(json.dumps(value));temp.replace(path)

def check_capacity(torch,phase):
    free,total=torch.cuda.mem_get_info();usable=(free+torch.cuda.memory_reserved())/2**20
    if usable<18000:raise RuntimeError('resource_capacity_changed: phase=%s free_plus_own_reserved_mib=%.1f required=18000'%(phase,usable))

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['directory','root','checkpoint']:parser.add_argument('--'+name,required=True)
    parser.add_argument('--backbone-directory');args=parser.parse_args();directory=Path(args.directory).resolve()
    try:
        profile=os.environ.get('GEOVERIFY_GEOMETRY_PROFILE','default')
        if profile not in ['default','normal9g','retained16g']:raise ValueError('Unknown geometry resource profile')
        import torch
        from geoverify.track4world_adapter import Track4WorldAdapter
        normal=profile=='normal9g';retained=profile=='retained16g';cap=9 if normal else (16 if retained else 0)
        if cap:torch.cuda.set_per_process_memory_fraction(cap*2**30/torch.cuda.get_device_properties(0).total_memory,0)
        if retained:
            check_capacity(torch,'before_model_load');reserve=torch.empty(15744*2**20,dtype=torch.uint8,device='cuda');del reserve
            torch.cuda.empty_cache=lambda:None
        model=Track4WorldAdapter(args.root,args.checkpoint,backbone_directory=args.backbone_directory)
        if retained:check_capacity(torch,'after_model_load')
        atomic_json(directory/'ready.json',dict(ready=True,resource_profile=profile));last_job=None
        while not (directory/'stop').exists():
            request_path=directory/'request.json'
            if not request_path.exists():time.sleep(.01);continue
            request=json.loads(request_path.read_text());job=request['job_id']
            if job==last_job:time.sleep(.01);continue
            rgb=np.load(directory/'frames.npy',mmap_mode='r',allow_pickle=False);queries=np.load(directory/'queries.npy',allow_pickle=False);validate_profile_window(profile,rgb.shape)
            from track4world.nets import blocks
            from contextlib import nullcontext
            from geoverify.correlation_pyramid import complete_correlation_pyramid
            from geoverify.depth_head_chunking import depth_head_chunks
            from geoverify.correlation_offload import offloaded_correlations
            high=rgb.shape[1]*rgb.shape[2]>=768*1024;bounded=normal or retained or high;chunk=2 if retained else 1
            if retained:check_capacity(torch,'before_window')
            with (offloaded_correlations(blocks) if bounded else complete_correlation_pyramid(blocks)),(depth_head_chunks(model.model,chunk) if bounded else nullcontext()):
                result=model.run_window(rgb,queries)
            result['resource_profile']=np.asarray(profile);result['resource_allocator_cap_gib']=np.asarray(cap);result['correlation_cpu_storage']=np.asarray(bounded);result['depth_head_frame_chunk_size']=np.asarray(chunk if bounded else 8);result['correlation_pyramid_protocol']=np.asarray('skip_unused_final_downsample_v1')
            if cap:
                result['resource_peak_reserved_mib']=np.asarray(torch.cuda.max_memory_reserved()/2**20);result['resource_peak_allocated_mib']=np.asarray(torch.cuda.max_memory_allocated()/2**20)
            if retained:result['retained_budget_mib']=np.asarray(15744)
            np.savez(directory/'result.pending.npz',**result);(directory/'result.pending.npz').replace(directory/'result.npz');atomic_json(directory/'response.json',dict(job_id=job,completed_monotonic=time.monotonic(),anchor_frame=request['anchor_frame'],target_frame=request['target_frame']));last_job=job
    except Exception:
        atomic_json(directory/'error.json',dict(error=traceback.format_exc()));raise
if __name__=='__main__':main()
