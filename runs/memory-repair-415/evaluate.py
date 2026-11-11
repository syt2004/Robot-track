"""Complete robot videos, every query cohort, actual async geometry and v13 output."""
from pathlib import Path
import sys,os,time,json,argparse,subprocess,hashlib,traceback
O=Path(__file__).resolve().parent;R=O.parents[1]
sys.path.insert(0,str(O/'source'));sys.path.insert(0,str(R/'runs/gpu-scheduling-recovery-051'))
from leases import snapshot,try_acquire
from geoverify.datasets import iter_evaluation
from geoverify.resource_profiles import profile_for_shape
p=argparse.ArgumentParser();p.add_argument('--checkpoint');p.add_argument('--baseline',action='store_true');p.add_argument('--output',required=True);p.add_argument('--indices',default='0,1,2,3,4,5,6,7,8,9,10,11');p.add_argument('--manifest',default=str(R/'runs/robot-development-expanded-002/manifest.json'));p.add_argument('--split',default='dev');args=p.parse_args()
D=Path(args.output).resolve();D.mkdir(exist_ok=False);lease=None;child=None;results=[]
try:
 source_manifest=json.loads((O/'source_manifest.json').read_text())
 for name,digest in source_manifest.items():assert hashlib.sha256((O/'source'/name).read_bytes()).hexdigest()==digest,name
 checkpoint=Path(args.checkpoint).resolve() if args.checkpoint else None
 if not args.baseline:assert checkpoint is not None and checkpoint.is_file()
 digest=hashlib.sha256(checkpoint.read_bytes()).hexdigest() if checkpoint else None
 manifest_path=Path(args.manifest).resolve();master=json.loads(manifest_path.read_text());selected=set(map(int,args.indices.split(',')))
 for i,s in enumerate(iter_evaluation(manifest_path,args.split)):
  if i not in selected:continue
  case=D/f'video_{i:02d}';case.mkdir(exist_ok=False);profile=profile_for_shape(s.rgb.shape[1:3]);need={'normal9g':19000,'retained16g':27000,'default':36000}[profile]
  if tuple(s.rgb.shape[1:3])==(128,128):
   old=R/'runs/memory-repair-413'
   probe=json.loads((old/'resource_probe_128.json').read_text());assert probe['state']=='passed' and probe['peak_reserved_mib']<8192
   for rel in ['geoverify/geometry_worker.py','geoverify/resource_profiles.py','geoverify/async_geometry.py']:assert (O/'source'/rel).read_bytes()==(old/'stable_source'/rel).read_bytes()
   need=24000
  while True:
   free,busy=snapshot();free={g:v for g,v in free.items() if g!=5};found=try_acquire({'shared':need},free,busy)
   if found:break
   (D/'status.json').write_text(json.dumps(dict(state='waiting_gpu',video=i,completed=len(results),need_mib=need),indent=2));time.sleep(15)
  chosen,lease=found;gpu=chosen['shared'];cfg=json.loads((O/'inference.json').read_text());
  if args.baseline:cfg.pop('memory_repair',None)
  else:cfg['memory_repair'].update(checkpoint=str(checkpoint),sha256=digest,audit_path=str(case/'memory_events.jsonl'),sequence_plan=str(R/'runs/memory-repair-413/sequence_plan.json'))
  cfg.update(geometry_physical_gpu=str(gpu),geometry_resource_profile=profile,ipc_root=str(case/'ipc'));cfg['experiment_scope']='Frozen v13 baseline without memory repair' if args.baseline else 'Actual complete-video memory repair evaluation; current checkpoint and source hashes bound.'
  (case/'inference.json').write_text(json.dumps(cfg,indent=2));manifest=dict(master);entry=dict(master['files'][i]);entry['path']=str((manifest_path.parent/entry['path']).resolve());manifest['files']=[entry]
  if 'selected_videos' in master:manifest['selected_videos']=[master['selected_videos'][i]]
  if manifest_path.parent.name=='robot-confirmation-106':
   manifest['original_role']=master.get('role');manifest['role']='Reused robot development evidence after413failure analysis; not untouched confirmation'
   manifest['source_manifest_sha256']=hashlib.sha256(manifest_path.read_bytes()).hexdigest()
  (case/'manifest.json').write_text(json.dumps(manifest,indent=2))
  env={**os.environ,'CUDA_VISIBLE_DEVICES':str(gpu),'PYTHONPATH':str(O/'source'),'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1','GEOVERIFY_GEOMETRY_PROFILE':profile,'MPLCONFIGDIR':'/tmp/memory413-eval-mpl'}
  start=time.monotonic()
  with (case/'inference.log').open('w') as f:
   child=subprocess.Popen([str(R/'.envs/trackon312/bin/python'),'-u',str(O/'frontend.py'),'--dataset-manifest',str(case/'manifest.json'),'--split',args.split,'--runtime-config',str(case/'inference.json'),'--output',str(case/'output'),'--no-overlays'],cwd=R,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
   while child.poll() is None:
    (D/'status.json').write_text(json.dumps(dict(state='running',video=i,name=s.name,completed=len(results),pid=child.pid,gpu=gpu,elapsed=time.monotonic()-start),indent=2));time.sleep(15)
   if child.returncode:raise RuntimeError(f'Video {i} failed; inspect {case}/inference.log')
  from audit_evaluation import audit
  audit(case,args.split)
  metrics=json.loads((case/'output/metrics.json').read_text());results.append(dict(index=i,video=s.name,metrics=metrics['metrics'],frames=len(s.rgb),queries=len(s.query_txy),wall_seconds=time.monotonic()-start));(D/'results.json').write_text(json.dumps(dict(checkpoint=str(checkpoint) if checkpoint else None,checkpoint_sha256=digest,baseline_only=args.baseline,results=results,mean_AJ=sum(v['metrics']['average_jaccard'] for v in results)/len(results),scope='Complete videos with actual async processing; compare historical baseline with timing caveat'),indent=2))
  lease.close();lease=None;child=None
  # Reclaim only completed worker input copies, never model outputs or source data.
  for directory in (case/'ipc').glob('geometry-*'):
   for filename in ('frames.npy','queries.npy'):
    path=directory/filename
    if path.exists():path.unlink()
 (D/'status.json').write_text(json.dumps(dict(state='complete',videos=len(results)),indent=2))
except BaseException:
 (D/'status.json').write_text(json.dumps(dict(state='failed',error=traceback.format_exc()),indent=2));raise
finally:
 if child and child.poll() is None:
  import signal
  os.killpg(child.pid,signal.SIGTERM)
  try:child.wait(timeout=10)
  except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL)
 if lease:lease.close()
