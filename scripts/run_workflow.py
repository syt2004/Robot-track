"""Execute the temporary DINOv2 full workflow after assets/environments are ready.

No automatic downloads/installations, fixture substitution or old run reuse.
Explicit prepared inputs replace preparation stages, preserving their provenance.
FPS inputs must state their source or assumption when preparing new datasets.
"""
import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import time


def commands(root, output, args):
    python=str(root/'.envs/trackon312/bin/python')
    geometry=str(root/'.envs/track4world311/bin/python')
    def module(name,*values):
        return [python,'-m',name,*map(str,values)]
    stages = [
        ('trackon_environment',module('geoverify.check_environment','--component','trackon','--root',root/'third_party/track_on')),
        ('geometry_environment',[geometry,'-m','geoverify.check_environment','--component','track4world','--root',str(root/'third_party/track4world')]),
        ('bootstrap_config',module('geoverify.configure_bootstrap','--root',root,'--output',output/'bootstrap.json','--geometry-physical-gpu',args.geometry_gpu,'--device','cuda:0')),
        ('kubric_prepare',module('geoverify.prepare_kubric','--selection',root/'configs/kubric.small.lock.json','--archive-root',root/'data/downloads/kubric','--output',output/'kubric','--fps',args.kubric_fps,'--fps-description',args.kubric_fps_description)),
        ('robotap_prepare',module('geoverify.prepare_robotap','--archive',root/'data/downloads/robotap.zip','--lock',root/'configs/datasets.lock.json','--output',output/'robotap','--fps',args.robotap_fps,'--fps-description',args.robotap_fps_description,'--videos',args.videos)),
        ('cache',module('geoverify.build_cache','--dataset-manifest',output/'kubric/manifest.json','--runtime-config',output/'bootstrap.json','--output',output/'cache')),
        ('train',[python,str(root/'train_calibrator.py'),'--manifest',str(output/'cache/manifest.json'),'--output',str(output/'training'),'--epochs',str(args.epochs),'--device','cuda:0']),
        ('promote',module('geoverify.promote_runtime','--bootstrap-config',output/'bootstrap.json','--checkpoint',output/'training/calibrator.pt','--output',output/'inference.json')),
        ('evaluate',module('geoverify.run_dataset','--dataset-manifest',output/'robotap/manifest.json','--split','eval','--runtime-config',output/'inference.json','--output',output/'evaluation')),
    ]
    # Explicit prepared manifests retain fixed query selections and FPS provenance.
    training=getattr(args,'training_manifest',None)
    evaluation=getattr(args,'evaluation_manifest',None)
    bootstrap=getattr(args,'bootstrap_config',None)
    if training: stages=[s for s in stages if s[0]!='kubric_prepare']
    if evaluation: stages=[s for s in stages if s[0]!='robotap_prepare']
    if bootstrap: stages=[s for s in stages if s[0]!='bootstrap_config']
    replacements={}
    if training: replacements[str(output/'kubric/manifest.json')]=str(training)
    if evaluation: replacements[str(output/'robotap/manifest.json')]=str(evaluation)
    if bootstrap: replacements[str(output/'bootstrap.json')]=str(bootstrap)
    stages=[(name,[replacements.get(value,value) for value in command]) for name,command in stages]
    for name,command in stages:
        if name=='bootstrap_config' and getattr(args,'metric_only',False): command.append('--metric-only')
        if name=='evaluate': command[command.index('--split')+1]=getattr(args,'evaluation_split','eval')
    return stages


def run(root,output,args):
    root=Path(root).resolve(strict=True)
    output=Path(output).resolve()
    for manifest_key,fps_key,description_key in (
        ('training_manifest','kubric_fps','kubric_fps_description'),
        ('evaluation_manifest','robotap_fps','robotap_fps_description')):
        path=getattr(args,manifest_key,None)
        if path:
            setattr(args,manifest_key,str(Path(path).resolve(strict=True)))
        else:
            fps=getattr(args,fps_key,None)
            if fps is None or not math.isfinite(fps) or fps<=0:
                raise ValueError('Positive finite FPS required for '+manifest_key)
            if not (getattr(args,description_key,None) or '').strip():
                raise ValueError('Explicit FPS provenance required for '+manifest_key)
    if args.epochs<1 or args.videos<1 or any(not str(g).isdigit() for g in (args.tracker_gpu,args.geometry_gpu)):
        raise ValueError('Positive epochs/videos and physical GPU indices required')
    if str(args.tracker_gpu)==str(args.geometry_gpu):
        raise ValueError('Tracker and geometry require separate physical GPUs')
    if getattr(args,'bootstrap_config',None):
        if getattr(args,'metric_only',False):
            raise ValueError('Existing bootstrap config already determines weight selection')
        args.bootstrap_config=str(Path(args.bootstrap_config).resolve(strict=True))
        config=json.loads(Path(args.bootstrap_config).read_text())
        if config.get('purpose')!='cache_bootstrap' or str(config.get('geometry_physical_gpu'))!=str(args.geometry_gpu):
            raise ValueError('Bootstrap purpose or physical geometry GPU mismatch')
    stages=commands(root,output,args)
    output.mkdir(parents=True,exist_ok=False)
    report=dict(state='running',variant='trackon2_dinov2',stages=[],
                tracker_physical_gpu=args.tracker_gpu,geometry_physical_gpu=args.geometry_gpu,
                acceptance='workflow execution only; accuracy and AR5E deployment remain separately verified')
    def save():
        temporary=output/'workflow.pending.json'
        temporary.write_text(json.dumps(report,indent=2)+'\n')
        temporary.replace(output/'workflow.json')
    save()
    try:
        for name,command in stages:
            entry=dict(stage=name,command=command,state='running')
            report['stages'].append(entry)
            save()
            print('Starting '+name,flush=True)
            environment=os.environ.copy()
            environment['CUDA_VISIBLE_DEVICES']=str(args.geometry_gpu if name=='geometry_environment' else args.tracker_gpu)
            start=time.monotonic()
            with (output/(name+'.log')).open('xb') as log:
                result=subprocess.run(command,cwd=root,env=environment,stdout=log,stderr=subprocess.STDOUT)
            entry.update(returncode=result.returncode,seconds=time.monotonic()-start,
                         state='complete' if result.returncode==0 else 'failed')
            save()
            if result.returncode:
                raise RuntimeError(name+' failed; inspect '+str(output/(name+'.log')))
        report['state']='complete'
        save()
    except BaseException as error:
        report.update(state='failed',error=type(error).__name__+': '+str(error))
        save()
        raise
    return output/'workflow.json'


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',default='.')
    parser.add_argument('--output',required=True)
    parser.add_argument('--tracker-gpu',required=True)
    parser.add_argument('--geometry-gpu',required=True)
    parser.add_argument('--kubric-fps',type=float)
    parser.add_argument('--kubric-fps-description')
    parser.add_argument('--robotap-fps',type=float)
    parser.add_argument('--robotap-fps-description')
    parser.add_argument('--training-manifest',help='Prepared Kubric manifest with fixed queries and timestamps')
    parser.add_argument('--evaluation-manifest',help='Prepared held-out manifest, including RoboTAP member subset')
    parser.add_argument('--evaluation-split',default='eval')
    parser.add_argument('--bootstrap-config',help='Bootstrap config with measured timing settings')
    parser.add_argument('--metric-only',action='store_true',help='Generate config for exact metric subset plus full T4W weights')
    parser.add_argument('--videos',type=int,default=3)
    parser.add_argument('--epochs',type=int,default=20)
    args=parser.parse_args()
    print(run(args.root,args.output,args))
