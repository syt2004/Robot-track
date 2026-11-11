"""Frozen-head fit on its own TRAIN cache; not independent video evaluation."""
import argparse,json,sys,math
from pathlib import Path
import numpy as np
import torch
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--experiment',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
a=p.parse_args();exp=a.experiment.resolve();sys.path.insert(0,str(exp/'source'))
from geoverify.calibration import GeometryCalibrationHead
from geoverify.datasets import file_sha256
from geoverify.point_sampling import labeled_rows
manifest=exp/'training_manifest.json';m=json.loads(manifest.read_text());checkpoint=exp/'training/calibrator.pt'
state=torch.load(checkpoint,map_location='cpu',weights_only=True)
assert state['manifest_sha256']==file_sha256(manifest)
for name,h in json.loads((exp/'source_manifest.json').read_text()).items():assert file_sha256(exp/'source'/name)==h
model=GeometryCalibrationHead().eval();model.load_state_dict(state['model'],strict=True);torch.set_num_threads(2)
rows={}
for shard in m['shards']:
 path=(manifest.parent/shard['path']).resolve();assert file_sha256(path)==shard['sha256']
 with np.load(path,allow_pickle=False) as archive:data=labeled_rows(dict(archive))
 # Keep grouped augmentation separate from the ungrouped same-scene input.
 variant='grouped' if 'grouped_cache' in str(path) else 'ungrouped'
 key=shard['scene']+' / '+variant
 row=rows.setdefault(key,dict(labeled_point_frames=0,visible_point_frames=0,raw_error_px_sum=0.,fused_error_px_sum=0.,raw_visibility_errors=0,fused_visibility_errors=0,raw_bce_sum=0.,fused_bce_sum=0.,raw_within_threshold_counts=[0]*5,fused_within_threshold_counts=[0]*5,raw_best_count=0,head_top1_best_count=0,head_top1_distance_px_sum=0.))
 for start in range(0,len(data['target_valid']),4096):
  d={k:torch.as_tensor(v[start:start+4096]) for k,v in data.items()}
  with torch.inference_mode():r=model(d['features'].float(),d['visual_logits'].float(),d['candidates'].float(),d['candidate_visibility'].float(),d['valid'])
  gt=d['target_visible'].bool();visible=gt & torch.isfinite(d['target_xy']).all(-1);n=len(gt);row['labeled_point_frames']+=n;row['visible_point_frames']+=int(visible.sum())
  geometry_weight=r['weights'][:,1:3].sum(-1)
  displacement=torch.linalg.vector_norm(r['xy']-d['candidates'][:,0],dim=-1)
  row['geometry_weight_sum']=row.get('geometry_weight_sum',0.)+float(geometry_weight.sum())
  row['geometry_top_choice_count']=row.get('geometry_top_choice_count',0)+int(((r['weights'].argmax(-1)==1)|(r['weights'].argmax(-1)==2)).sum())
  row['position_changed_over_1px_count']=row.get('position_changed_over_1px_count',0)+int((displacement>1).sum())
  row['position_changed_over_0_1px_count']=row.get('position_changed_over_0_1px_count',0)+int((displacement>.1).sum())
  rawvis=d['candidate_visibility'][:,0].float();fusedvis=r['visibility']
  for name,vis in [('raw',rawvis),('fused',fusedvis)]:
   row[name+'_visibility_errors']+=int(((vis>=.5)!=gt).sum())
   row[name+'_bce_sum']+=float(torch.nn.functional.binary_cross_entropy(vis.clamp(1e-6,1-1e-6),gt.float(),reduction='sum'))
  if not bool(visible.any()):continue
  scale=torch.tensor([256/shard['width'],256/shard['height']])
  for name,xy in [('raw',d['candidates'][:,0]),('fused',r['xy'])]:
   delta=xy[visible]-d['target_xy'][visible]
   row[name+'_error_px_sum']+=float(torch.linalg.vector_norm(delta,dim=-1).sum())
   dist=torch.linalg.vector_norm(delta*scale,dim=-1)
   row[name+'_within_threshold_counts']=[c+int((dist<t).sum()) for c,t in zip(row[name+'_within_threshold_counts'],[1,2,4,8,16])]
  distance=torch.linalg.vector_norm(d['candidates'][visible]-d['target_xy'][visible,None,:],dim=-1).masked_fill(~d['valid'][visible].bool(),torch.inf)
  best=distance.argmin(-1);top=r['weights'][visible].argmax(-1)
  chosen=d['candidates'][visible].gather(1,top[:,None,None].expand(-1,1,2)).squeeze(1)
  selected_distance=torch.linalg.vector_norm((chosen-d['target_xy'][visible])*scale,dim=-1)
  counts=row.setdefault('head_top1_within_threshold_counts',[0]*5)
  row['head_top1_within_threshold_counts']=[c+int((selected_distance<t).sum()) for c,t in zip(counts,[1,2,4,8,16])]
  fused_distance=torch.linalg.vector_norm(r['xy'][visible]-d['target_xy'][visible],dim=-1)
  row['fused_better_than_every_candidate']=row.get('fused_better_than_every_candidate',0)+int((fused_distance+1e-5<distance.min(-1).values).sum())
  # Local logit-space directions only: not a parameter-space training step.
  w=r['weights'][visible];valid=d['valid'][visible].bool()
  clean=torch.where(valid[...,None],d['candidates'][visible],0.)
  diagonal=math.hypot(shard['width'],shard['height'])
  grad_xy=((r['xy'][visible]-d['target_xy'][visible])/diagonal/.01).clamp(-1,1)/diagonal
  grad_track=w*((clean-r['xy'][visible,None,:])*grad_xy[:,None,:]).sum(-1)
  grad_rank=w-torch.nn.functional.one_hot(best,num_classes=w.shape[-1])
  # smooth_l1_loss averages the TWO coordinate components; account for that
  # and the actual loss coefficient 10. Both losses average the same rows.
  track_norm=torch.linalg.vector_norm(5*grad_track,dim=-1)
  rank_norm=torch.linalg.vector_norm(grad_rank,dim=-1)
  row['rank_logit_gradient_norm_sum']=row.get('rank_logit_gradient_norm_sum',0.)+float(rank_norm.sum())
  row['weighted_track_logit_gradient_norm_sum']=row.get('weighted_track_logit_gradient_norm_sum',0.)+float(track_norm.sum())
  row['rank_gradient_over_10x_track_rows']=row.get('rank_gradient_over_10x_track_rows',0)+int((rank_norm>10*track_norm).sum())
  dot=(grad_track*grad_rank).sum(-1)
  row['rank_vs_position_logit_gradient_conflicts']=row.get('rank_vs_position_logit_gradient_conflicts',0)+int((dot < -1e-12).sum())
  row['raw_best_count']+=int((best==0).sum());row['head_top1_best_count']+=int((top==best).sum())
  row['head_top1_distance_px_sum']+=float(distance.gather(-1,top[:,None]).sum())
for r in rows.values():
 r['head_top1_average_position_threshold_accuracy']=sum(r['head_top1_within_threshold_counts'])/(5*r['visible_point_frames'])
 for name in ['raw','fused']:
  r[name+'_mean_error_pixels']=r[name+'_error_px_sum']/r['visible_point_frames']
  r[name+'_visibility_accuracy']=1-r[name+'_visibility_errors']/r['labeled_point_frames']
  r[name+'_mean_visibility_bce']=r[name+'_bce_sum']/r['labeled_point_frames']
  r[name+'_average_position_threshold_accuracy']=sum(r[name+'_within_threshold_counts'])/(5*r['visible_point_frames'])
report=dict(state='complete',checkpoint_sha256=file_sha256(checkpoint),manifest_sha256=file_sha256(manifest),rows=rows,gradient_scope='Per-row logit gradients only, not shared-parameter gradients or an actual optimizer step. Sign tolerance 1e-12; distance tolerance 1e-5 px.',scope='Own TRAIN cache fit only, not independent online/video evaluation. Cache feedback is bootstrap-generated. All labeled rows, query frames excluded; grouped same-scene augmentation separated. Threshold position accuracy uses 256-coordinate scaling, no visibility prediction. No model changes or fitted thresholds.')
a.output.mkdir(parents=True,exist_ok=False);(a.output/'audit.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
