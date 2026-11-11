"""Wait for published model assets, then run real geometry latency diagnostics."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',required=True);p.add_argument('--gpu',type=int,required=True)
    p.add_argument('--dataset-manifest',default='data/prepared/kubric-development-256/manifest.json')
    a=p.parse_args()
    if a.gpu<0:p.error('Nonnegative GPU required')
    root=Path(__file__).resolve().parents[1];out=Path(a.output).resolve();out.mkdir(parents=True,exist_ok=False)
    def status(stage,**extra):
        record=dict(stage=stage,**extra)
        tmp=out/'status.pending.json';tmp.write_text(json.dumps(record,indent=2));tmp.replace(out/'status.json')
        print(json.dumps(record),flush=True)
    deadline=time.monotonic()+24*3600
    inputs=[root/'weights/track4world/track4world_da3.pth',root/'weights/da3nested-metric/metric_source.json']
    try:
        status('waiting_for_published_weights')
        while not all(f.is_file() for f in inputs):
            if time.monotonic()>deadline:raise TimeoutError('Weight wait exceeded 24 hours')
            time.sleep(10)
        status('waiting_for_idle_gpu',physical_gpu=a.gpu)
        while True:
            result=subprocess.run(['nvidia-smi','-i',str(a.gpu),'--query-gpu=memory.used,utilization.gpu',
                                   '--format=csv,noheader,nounits'],check=True,capture_output=True,text=True)
            memory,util=map(int,result.stdout.strip().split(','))
            if memory<1024 and util==0:break
            if time.monotonic()>deadline:raise TimeoutError('GPU wait exceeded deadline')
            time.sleep(10)
        env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=str(a.gpu),OMP_NUM_THREADS='1',
                                        HF_HUB_OFFLINE='1',MPLCONFIGDIR='/tmp/tracking-mpl')
        command=[str(root/'.envs/track4world311/bin/python'),'-u','-m','geoverify.probe_geometry',
                 '--dataset-manifest',a.dataset_manifest,
                 '--output',str(out/'probe'),'--metric-only','--windows','2','2','4','8','16']
        status('geometry_probe_running',command=command)
        with (out/'probe.log').open('x') as log:
            subprocess.run(command,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        status('complete',full_pipeline=False,probe=str(out/'probe/status.json'))
    except BaseException as error:
        status('failed',error=type(error).__name__+': '+str(error));raise


if __name__=='__main__':main()
