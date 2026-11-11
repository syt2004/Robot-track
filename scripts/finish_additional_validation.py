"""Finish the fixed extra validation subset when its owned download publishes files.

Does not start/restart a downloader. Validates archives in prepare_kubric, then
waits for an idle explicitly selected GPU before frozen baseline evaluation.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True)
    parser.add_argument('--gpu',required=True,type=int)
    parser.add_argument('--dataset',choices=('kubric','robotap'),default='kubric')
    args=parser.parse_args()
    if args.gpu<0:parser.error('GPU must be nonnegative')
    root=Path(__file__).resolve().parents[1]
    out=Path(args.output).resolve();out.mkdir(parents=True,exist_ok=False)
    lock=json.loads((root/'configs/kubric.additional-val.lock.json').read_text())
    inputs=([root/'data/downloads/kubric'/e['path'] for e in lock['files']] if args.dataset=='kubric' else
            [root/'data/downloads'/name for name in ('robotap-split4-prefix.bin','robotap_split4.pkl','robotap_split4.provenance.json')])
    python=str(root/'.envs/trackon312/bin/python')
    prepared=out/'prepared';selected=out/'selected/manifest.json'
    deadline=time.monotonic()+24*3600
    def status(stage,**kwargs):
        value=dict(stage=stage,dataset=args.dataset,**kwargs)
        temp=out/'status.pending.json';temp.write_text(json.dumps(value,indent=2));temp.replace(out/'status.json')
        print(json.dumps(value),flush=True)
    def invoke(stage,command,env=None):
        status(stage,command=command)
        with (out/(stage+'.log')).open('x') as stream:
            subprocess.run(command,cwd=root,stdout=stream,stderr=subprocess.STDOUT,env=env,check=True)
    try:
        status('waiting_for_verified_archive_publication')
        while not all(path.is_file() for path in inputs):
            if time.monotonic()>deadline:raise TimeoutError('Archive publication wait exceeded 24 hours')
            time.sleep(10)
        if args.dataset=='kubric':
            invoke('prepare',[python,'-m','geoverify.prepare_kubric','--selection','configs/kubric.additional-val.lock.json',
                              '--output',str(prepared),'--fps','24','--fps-description','Assumed 24 Hz replay; source FPS unknown'])
            invoke('select',[python,'-m','geoverify.select_kubric_queries','--source',str(prepared/'manifest.json'),
                             '--output',str(selected),'--count','256','--seed','20261030'])
        else:
            invoke('prepare',[python,'-m','geoverify.prepare_robotap_member','--output',str(prepared),
                              '--fps','24','--fps-description','Assumed 24 Hz development replay; not measured source FPS',
                              '--videos','3','--seed','20261029'])
            selected=prepared/'manifest.json'
        import os
        env=os.environ.copy();env.update(OMP_NUM_THREADS='1',MPLCONFIGDIR='/tmp/tracking-mpl',HF_HUB_OFFLINE='1',
                                        CUDA_VISIBLE_DEVICES=str(args.gpu))
        if args.dataset=='kubric':
            invoke('coverage',[python,'-m','geoverify.audit_kubric_subset','--dataset-manifest',str(selected),
                               '--output',str(out/'coverage')],env)
        status('waiting_for_idle_gpu',physical_gpu=args.gpu)
        while True:
            query=subprocess.run(['nvidia-smi','-i',str(args.gpu),'--query-gpu=memory.used,utilization.gpu',
                                  '--format=csv,noheader,nounits'],capture_output=True,text=True,check=True)
            used,util=map(int,query.stdout.strip().split(','))
            if used<1024 and util==0:break
            if time.monotonic()>deadline:raise TimeoutError('Idle GPU wait exceeded deadline')
            time.sleep(10)
        invoke('baseline',[python,'-u','-m','geoverify.run_trackon_baseline','--dataset-manifest',str(selected),
                           '--split','val' if args.dataset=='kubric' else 'eval','--output',str(out/'baseline')],env)
        status('complete',full_fusion=False,dataset_manifest=str(selected),baseline=str(out/'baseline'))
    except BaseException as error:
        status('failed',error=type(error).__name__+': '+str(error));raise


if __name__=='__main__':main()
