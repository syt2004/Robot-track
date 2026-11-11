"""Independent-shadow memory correction trained through sixteen future frames."""
from pathlib import Path
import sys,os,json,time,hashlib,argparse,traceback
R=Path(__file__).resolve().parents[2];O=Path(__file__).resolve().parent;DATA=R/'runs/memory-repair-413'
sys.path.insert(0,str(O/'source'));sys.path.insert(0,str(R/'runs/gpu-scheduling-recovery-051'))
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
os.environ['NUMPY_MADVISE_HUGEPAGE']='0'
a=argparse.ArgumentParser();a.add_argument('--seed',type=int,default=0);a.add_argument('--epochs',type=int,default=3);a.add_argument('--output',required=True);args=a.parse_args()
D=Path(args.output);D.mkdir(exist_ok=False);lease=None
try:
 from leases import snapshot,try_acquire
 while True:
  status=O/'preparation_status.json'
  if status.exists():
   state=json.loads(status.read_text())
   if state['state']=='failed':raise RuntimeError('Sequence preparation failed; inspect its status')
   if state['state']=='complete':break
  (D/'status.json').write_text(json.dumps(dict(state='waiting_sequence_preparation')));time.sleep(15)
 while True:
  free,busy=snapshot();free={g:v for g,v in free.items() if g!=5};found=try_acquire({'train':10000},free,busy)
  if found:break
  (D/'status.json').write_text(json.dumps(dict(state='waiting_gpu',free_mib=free)));time.sleep(15)
 chosen,lease=found;os.environ['CUDA_VISIBLE_DEVICES']=str(chosen['train'])
 import torch,numpy as np
 import torch.nn.functional as F
 from PIL import Image
 from geoverify.adapters import TrackOnAdapter
 from geoverify.memory_repair import GeometryMemoryRepair,bound_to_reference
 from geoverify.memory_training import track_frame_with_grad
 torch.set_num_threads(2);torch.manual_seed(args.seed);np.random.seed(args.seed);torch.cuda.set_per_process_memory_fraction(.22)
 c=json.loads((DATA/'inference.json').read_text());base=TrackOnAdapter(c['trackon_root'],c['trackon_checkpoint'],variant='trackon2_dinov2',backbone_directory=c['backbone_directory']);m=base.predictor.model
 plan=json.loads((DATA/'sequence_plan.json').read_text());val=set(plan['validation_sources']);long_targets=torch.load(DATA/'long_rollout_targets.pt',map_location='cpu',weights_only=True)
 paths=sorted((DATA/'sequence_cache').glob('*.pt'));assert len(paths)==417
 verified_manifest=json.loads((O/'verified_manifest.json').read_text())['files'];assert len(verified_manifest)==417
 fit=[p for p in paths if p.stem.split('_')[0] not in val];validation=[p for p in paths if p.stem.split('_')[0] in val]
 repair=GeometryMemoryRepair(m.D).cuda();optimizer=torch.optim.AdamW(repair.parameters(),lr=1e-4,weight_decay=1e-4)
 checked=set();rng=np.random.default_rng(args.seed);history=[];best=-float('inf');start=time.monotonic()
 def episode(path,train):
  ep=torch.load(path,map_location='cpu',weights_only=True);ep.update(long_targets[path.stem]);memory=ep['memory'].cuda();sampled=torch.zeros_like(memory);evidence=memory.new_zeros((*memory.shape[:2],8));valid=torch.zeros_like(ep['mask'],device='cuda');slots=ep['slots']
  vp=O/'verified_cache'/path.name
  assert hashlib.sha256(vp.read_bytes()).hexdigest()==verified_manifest[path.name]
  verified=torch.load(vp,map_location='cpu',weights_only=True)
  assert verified['episode_sha256']==hashlib.sha256(path.read_bytes()).hexdigest()
  sampled[:,slots]=ep['sampled'].cuda();ev=ep['evidence'].clone();ev[:,:,-2:]=verified['features'].permute(1,0,2);evidence[:,slots]=ev.cuda();valid[:,slots]=(ep['valid'] & verified['accept'].T).cuda()
  q=ep['q'].cuda();mask=ep['mask'].cuda();shadow=memory.clone();mem=repair(memory,sampled,evidence,valid);relative=(mem-memory).norm(dim=-1)/memory.norm(dim=-1).clamp_min(1e-6);loss=.1*relative[valid].square().mean() if valid.any() else mem.sum()*0;errors=[];correct=[];positions=[];predvis=[]
  for t,entry in enumerate(ep['frames']):
   fp=Path(entry['path'])
   if str(fp) not in checked:
    assert hashlib.sha256(fp.read_bytes()).hexdigest()==entry['sha256'];checked.add(str(fp))
   rgb=np.asarray(Image.open(fp).convert('RGB'));frame=torch.tensor(rgb.copy(),device='cuda').permute(2,0,1)[None].float()
   features=m.extract_frame_features(frame)
   with torch.no_grad():_,_,shadow_new=m.track_frame(q,mask,shadow,features,ep['h'],ep['w'])
   out=track_frame_with_grad(m,q,mask,mem,features,ep['h'],ep['w']);target=ep['target'][t].cuda();vis=ep['visible'][t].cuda();positions.append(out['xy'].detach().cpu());predvis.append((out['visibility_logit'].sigmoid()>=.5).detach().cpu())
   coords=target*target.new_tensor([m.W/ep['w'],m.H/ep['h']]);patch=(coords[:,1]/m.stride).long()*m.Wf+(coords[:,0]/m.stride).long()
   loss=loss+F.binary_cross_entropy_with_logits(out['visibility_logit'],vis.float())/len(ep['frames'])
   if vis.any():
    loss=loss+3*F.cross_entropy(out['correlation_logits'][vis]/.05,patch[vis].clamp(0,m.P-1))/len(ep['frames'])
    native_error=(out['xy'][vis]-target[vis])*target.new_tensor([m.W/ep['w'],m.H/ep['h']])
    loss=loss+native_error.abs().sum(-1).clamp(max=2*m.stride).mean()/len(ep['frames'])
    err=torch.linalg.vector_norm((out['xy'][vis]-target[vis])*target.new_tensor([256/ep['w'],256/ep['h']]),dim=-1)
    errors.append(float(err.mean().detach()));correct.append(float((err<8).float().mean()))
   bounded_new=bound_to_reference(out['q_new'],shadow_new,.25)
   mem=torch.cat((mem[:,1:],bounded_new[:,None]),1);shadow=torch.cat((shadow[:,1:],shadow_new[:,None]),1);mask=torch.cat((mask[:,1:],torch.zeros(len(q),1,dtype=torch.bool,device='cuda')),1)
  if isinstance(loss,int):return None
  if train:
   optimizer.zero_grad();loss.backward();gn=torch.nn.utils.clip_grad_norm_(repair.parameters(),1.);optimizer.step()
   if not torch.isfinite(gn):raise ValueError('Nonfinite gradient')
  if not errors:return None
  pred=torch.stack(positions);visible=torch.stack(predvis);gtvis=ep['visible'];distance=torch.linalg.vector_norm((pred-ep['target'])*torch.tensor([256/ep['w'],256/ep['h']]),dim=-1)
  aj=[];apw=[]
  for threshold in (1,2,4,8,16):
   hit=distance<threshold;tp=(hit&gtvis&visible).sum();fp=((~hit|~gtvis)&visible).sum();positive=gtvis.sum()
   aj.append(float(tp/(positive+fp)));apw.append(float((hit&gtvis).sum()/positive))
  return [float(loss.detach()),float(np.mean(errors)),float(np.mean(correct)),float(np.mean(aj)),float(np.mean(apw))]
 with torch.no_grad():
  identity=np.mean([episode(path,False) for path in validation],axis=0).tolist()
 (D/'identity_validation.json').write_text(json.dumps(dict(metrics=identity,metric_names=['loss','EPE256','within8','AJ','APW']),indent=2));print('identity validation',identity,flush=True)
 for epoch in range(args.epochs):
  fitrows=[]
  for i in rng.permutation(len(fit)):
   row=episode(fit[i],True)
   if row is not None:fitrows.append(row)
   if len(fitrows)%20==0:
    (D/'status.json').write_text(json.dumps(dict(state='training',epoch=epoch,events=len(fitrows),total=len(fit),elapsed=time.monotonic()-start),indent=2))
  valrows=[]
  with torch.no_grad():
   for path in validation:
    row=episode(path,False)
    if row is not None:valrows.append(row)
  row=dict(epoch=epoch,train=np.mean(fitrows,axis=0).tolist(),validation=np.mean(valrows,axis=0).tolist(),elapsed=time.monotonic()-start);history.append(row);print(row,flush=True)
  state=dict(model=repair.state_dict(),dimension=m.D,epoch=epoch,seed=args.seed,train_sources=plan['train_sources'],validation_sources=plan['validation_sources'],metrics=row,architecture='verified_geometry_shadow_memory_v4',maximum_relative_change=.25,long_rollout_targets_sha256=hashlib.sha256((DATA/'long_rollout_targets.pt').read_bytes()).hexdigest(),verified_manifest_sha256=hashlib.sha256((O/'verified_manifest.json').read_bytes()).hexdigest(),scope='Historical appearance-verified writes; 111 sources fit /28 validation; three single-event sixteen-frame rollouts per source; independent shadow reference, bounded writes and recurrent transitions; backbones frozen',objective=dict(temperature=.05,patch_weight=3.,position='native-input-pixel L1 capped at 2*stride',visibility='BCE',selection='maximum sixteen-frame validation AJ',relative_memory_penalty=.1,reference='independent_unmodified_shadow',transition_relative_bound=.25,write_admission='positive paired historical verifier margin with numerical tie tolerance1e-6'),metric_names=['loss','EPE256','within8','AJ','APW'],plan_sha256=hashlib.sha256((DATA/'sequence_plan.json').read_bytes()).hexdigest())
  torch.save(state,D/'last.pending');(D/'last.pending').replace(D/'last.pt')
  if row['validation'][3]>best:
   best=row['validation'][3];torch.save(state,D/'best.pending');(D/'best.pending').replace(D/'best.pt')
  (D/'history.json').write_text(json.dumps(history,indent=2))
 (D/'status.json').write_text(json.dumps(dict(state='complete',best_validation_AJ=best,elapsed=time.monotonic()-start),indent=2))
except BaseException:
 (D/'status.json').write_text(json.dumps(dict(state='failed',error=traceback.format_exc()),indent=2));raise
finally:
 if lease:lease.close()
