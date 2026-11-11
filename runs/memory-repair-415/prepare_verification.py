"""Verify existing417 synthetic geometry windows; no downloads or robot labels."""
from pathlib import Path
import os,sys,json,hashlib,time,traceback,argparse
O=Path(__file__).resolve().parent;R=O.parents[1];DATA=R/'runs/memory-repair-413'
sys.path.insert(0,str(O/'source'));sys.path.insert(0,str(R/'runs/gpu-scheduling-recovery-051'))
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
os.environ['NUMPY_MADVISE_HUGEPAGE']='0'
p=argparse.ArgumentParser();p.add_argument('--limit',type=int);a=p.parse_args();D=O/'verified_cache';D.mkdir(exist_ok=True);lease=None
try:
 from leases import snapshot,try_acquire
 while True:
  free,busy=snapshot();free={k:v for k,v in free.items() if k!=5};found=try_acquire({'prepare':8000},free,busy)
  if found:break
  (O/'preparation_status.json').write_text(json.dumps(dict(state='waiting_gpu')));time.sleep(15)
 chosen,lease=found;os.environ['CUDA_VISIBLE_DEVICES']=str(chosen['prepare'])
 import torch,numpy as np
 from PIL import Image
 from geoverify.adapters import VerifierAdapter
 from geoverify.verified_write import verify_historical_window,admission_from_scores
 torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.14)
 c=json.loads((O/'inference.json').read_text());v=VerifierAdapter(c['trackon_root'],c['verifier_checkpoint']);t=time.monotonic()
 coordinate=json.loads((DATA/'coordinate_manifest.json').read_text())['files'];episodes=json.loads((DATA/'sequence_cache_manifest.json').read_text())['files'];counts=[]
 paths=sorted((DATA/'sequence_cache').glob('*.pt'));assert len(paths)==417
 for i,path in enumerate(paths[:a.limit] if a.limit else paths):
  destination=D/path.name
  if destination.exists():continue
  assert hashlib.sha256(path.read_bytes()).hexdigest()==episodes[path.name]
  cp=DATA/'coordinate_cache'/path.name;assert hashlib.sha256(cp.read_bytes()).hexdigest()==coordinate[path.name]
  ep=torch.load(path,map_location='cpu',weights_only=True);xy=torch.load(cp,map_location='cpu',weights_only=True)
  first=Path(ep['frames'][0]['path']);anchor,end=ep['geometry']['anchor'],ep['geometry']['target']
  frame_paths=[first.parent/f'{n:03d}.png' for n in range(anchor,end+1)]
  rgb=np.stack([np.asarray(Image.open(f).convert('RGB')) for f in frame_paths]);ref=xy['raw_xy'].numpy().transpose(1,0,2);geo=xy['geometry_xy_safe'].numpy().transpose(1,0,2);valid=ep['valid'].numpy().T
  if i==0:
   # Equal reference/geometry must have identical scores and no writes.
   identical=verify_historical_window(v,rgb,ref,ref,np.ones_like(valid));assert not identical['accept'].any()
   candidates=np.stack((ref,geo),2);availability=np.stack((np.ones_like(valid),valid),2);query=np.c_[np.zeros(ref.shape[1]),ref[0]].astype(np.float32)
   expected=v.rank_window(rgb,query,candidates,availability)
  out=verify_historical_window(v,rgb,ref,geo,valid)
  if i==0:
   np.testing.assert_allclose(out['scores'][-1][availability[-1]],expected[availability[-1]],atol=1e-5,rtol=1e-6)
   (O/'verification_checks.json').write_text(json.dumps(dict(state='passed',identical_candidates_accept=0,last_frame_original_adapter_max_diff=float(np.abs(out['scores'][-1][availability[-1]]-expected[availability[-1]]).max()),scope='Real frozen Verifier; historical all-frame scores agree with original last-frame adapter'),indent=2))
  record={k:torch.from_numpy(x.copy()) for k,x in out.items()};record.update(source=ep['source'],event=ep['event'],episode_sha256=episodes[path.name],coordinate_sha256=coordinate[path.name],rgb_sha256={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in frame_paths},verifier_sha256=c['verifier_sha256'] if 'verifier_sha256' in c else hashlib.sha256(Path(c['verifier_checkpoint']).read_bytes()).hexdigest())
  temp=destination.with_suffix('.pending');torch.save(record,temp);temp.replace(destination)
  counts.append((int(out['accept'].sum()),int(out['valid'].sum())))
  state=dict(state='preparing',completed=len(list(D.glob('*.pt'))),total=417,gpu=chosen['prepare'],elapsed=time.monotonic()-t);(O/'preparation_status.json').write_text(json.dumps(state));
  if i%20==0:print(state,flush=True)
 files={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(D.glob('*.pt'))}
 if not a.limit:assert len(files)==417
 (O/'verified_manifest.json').write_text(json.dumps(dict(files=files,protocol='paired_historical_verifier_margin_v1',source=__doc__),indent=2))
 (O/'preparation_status.json').write_text(json.dumps(dict(state='complete' if len(files)==417 else 'partial',events=len(files),elapsed=time.monotonic()-t,accepted=sum(x[0] for x in counts),valid=sum(x[1] for x in counts)),indent=2))
except Exception:
 (O/'preparation_status.json').write_text(json.dumps(dict(state='failed',error=traceback.format_exc()),indent=2));raise
finally:
 if lease:lease.close()
