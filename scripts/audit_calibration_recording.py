"""Recompute saved calibration inputs with the exact experiment checkpoint on CPU.

No GT or threshold fitting. This verifies the calibration stage, not frontend,
geometry quality, causal feature construction, or whole-pipeline accuracy.
"""
import argparse,json,hashlib,sys
from pathlib import Path
import numpy as np
import torch

def digest(path):
 h=hashlib.sha256()
 with open(path,'rb') as f:
  for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
 return h.hexdigest()

def main():
 parser=argparse.ArgumentParser(description=__doc__)
 parser.add_argument('--experiment',required=True);parser.add_argument('--tracks',required=True)
 parser.add_argument('--config',default='inference.json');parser.add_argument('--output',required=True)
 args=parser.parse_args();exp=Path(args.experiment).resolve();track=Path(args.tracks).resolve();output=Path(args.output).resolve()
 if output.exists():raise FileExistsError(output)
 src=exp/'source'
 for name,h in json.loads((exp/'source_manifest.json').read_text()).items():
  assert digest(src/name)==h,'Source changed: '+name
 sys.path.insert(0,str(src))
 from geoverify.calibration import GeometryCalibrationHead,HEAD_ARCHITECTURE
 config=json.loads((exp/args.config).read_text());checkpoint=Path(config['calibrator_checkpoint'])
 assert digest(checkpoint)==config['calibrator_sha256'],'Checkpoint binding mismatch'
 manifest_bound=False
 progress=track.parent.parent/'progress.json'
 if progress.exists():
  recorded=json.loads(progress.read_text());entries=[v for v in recorded['videos'] if (progress.parent/v['path']).resolve()==track]
  assert len(entries)==1 and entries[0]['sha256']==digest(track),'Prediction manifest binding mismatch'
  assert recorded['run_metadata']['calibrator_sha256']==digest(checkpoint)
  assert recorded['run_metadata']['runtime_config_sha256']==digest(exp/args.config)
  assert digest(progress.parent/'runtime_config.snapshot.json')==digest(exp/args.config)
  manifest_bound=True
 state=torch.load(checkpoint,map_location='cpu',weights_only=True)
 assert state['head_architecture']==config['head_architecture']==HEAD_ARCHITECTURE
 assert state.get('evidence_age_protocol')==config.get('evidence_age_protocol'),'Clock protocol binding mismatch'
 assert state['manifest_sha256']==digest(exp/'training_manifest.json')
 model=GeometryCalibrationHead().eval();model.load_state_dict(state['model'],strict=True)
 a=dict(np.load(track,allow_pickle=False));mask=a['processed'].astype(bool)
 np.testing.assert_array_equal(a['calibration_inputs_recorded'],mask)
 assert mask.any();valid=a['candidate_valid'][mask].astype(bool)
 keys=['calibration_features','visual_logits','candidates','candidate_visibility','candidate_valid']
 flat={k:a[k][mask] for k in keys};maxdiff={k:0. for k in ['xy','visibility','confidence','weights']};counterfactual_max=0.;no_geometry_rows=0
 with torch.inference_mode():
  for start in range(0,int(mask.sum()),2048):
   end=start+2048
   inputs=[torch.as_tensor(flat[k][start:end]).bool() if k=='candidate_valid' else torch.as_tensor(flat[k][start:end]).float() for k in keys]
   result=model(*inputs)
   for key,saved in [('xy','xy'),('visibility','visibility'),('confidence','confidence'),('weights','fusion_weights')]:
    expected=a[saved][mask][start:end];actual=result[key].numpy();diff=float(np.max(np.abs(actual-expected)));maxdiff[key]=max(maxdiff[key],diff)
    np.testing.assert_allclose(actual,expected,atol=2e-3 if key=='xy' else 1e-5,rtol=1e-6,err_msg='Saved '+key+' cannot be recomputed')
   if HEAD_ARCHITECTURE in ('candidate_rank_current_visibility_valid_age_v3',
                           'candidate_rank_current_visibility_valid_age_position1000_v5'):
    absent=~inputs[-1][...,1:3].any(-1)
    if absent.any():
     no_geometry_rows+=int(absent.sum());low=[v[absent].clone() for v in inputs];high=[v.clone() for v in low];low[0][...,11]=0.;high[0][...,11]=10000.
     low_out=model(*low);high_out=model(*high)
     for key in maxdiff:
      delta=float((low_out[key]-high_out[key]).abs().max());counterfactual_max=max(counterfactual_max,delta);assert delta==0.,'Expired age affects '+key
 np.testing.assert_array_equal(a['visible'],(a['visibility']>=float(a['visibility_threshold']))&mask)
 report=dict(state='passed',prediction_manifest_bound=manifest_bound,tracks_sha256=digest(track),checkpoint_sha256=digest(checkpoint),head_architecture=HEAD_ARCHITECTURE,processed_point_frames=int(mask.sum()),max_abs_cpu_saved_difference=maxdiff,no_geometry_age_invariance_rows=no_geometry_rows,no_geometry_age_invariance_max_difference=counterfactual_max if no_geometry_rows else None,scope='Exact calibration input/output replay only. CPU/GPU tolerance: xy atol .002 pixels, other outputs 1e-5, rtol 1e-6. No GT and no accuracy claim.')
 output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
