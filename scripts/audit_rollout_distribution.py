"""Same TRAIN video: bootstrap cache versus learned-feedback recording.

Independent asynchronous geometry makes this a distribution diagnostic, not a
controlled causal estimate of feedback alone. No training or threshold fitting.
"""
import argparse, json, sys
from pathlib import Path
import numpy as np
import torch

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--experiment', required=True, type=Path)
args=p.parse_args(); exp=args.experiment.resolve()
sys.path.insert(0,str(exp/'source'))
from geoverify.datasets import file_sha256, iter_evaluation
from geoverify.calibration import GeometryCalibrationHead, canonical_features
from geoverify.feature_schema import FEATURE_NAMES
status=json.loads((exp/'status.json').read_text())
if status['state']!='complete':
    recovery=json.loads((exp/'recovery_audit.json').read_text())
    assert recovery['state']=='complete' and recovery['original_state']==status
    assert file_sha256(exp/'tracks.npz')==recovery['tracks_sha256']
binding=json.loads((exp/'binding.json').read_text())
assert file_sha256(exp/'tracks.npz')==binding['tracks_sha256']
for name,h in json.loads((exp/'source_manifest.json').read_text()).items():
    assert file_sha256(exp/'source'/name)==h
contract=json.loads((exp/'experiment.json').read_text())
sample=next(iter_evaluation(contract['dataset_manifest'],'train'))
assert sample.name==binding['video'] and sample.source_sha256==binding['source_sha256']
config=json.loads((exp/'inference.json').read_text())
assert file_sha256(config['calibrator_checkpoint'])==binding['checkpoint_sha256']==config['calibrator_sha256']
state=torch.load(config['calibrator_checkpoint'],map_location='cpu',weights_only=True)
assert state['manifest_sha256']==file_sha256(contract['training_manifest'])
manifest=json.loads(Path(contract['training_manifest']).read_text())
model=GeometryCalibrationHead().eval();model.load_state_dict(state['model'],strict=True);torch.set_num_threads(2)
a=dict(np.load(exp/'tracks.npz',allow_pickle=False)); seen=np.zeros(a['processed'].shape,bool)
old={k:[] for k in ['features','visual_logits','candidates','candidate_visibility','valid']}
new={k:[] for k in old};targets=[];visibility=[];saved=[]
mapping=dict(features='calibration_features',visual_logits='visual_logits',candidates='candidates',candidate_visibility='candidate_visibility',valid='candidate_valid')
used=[]
for shard in manifest['shards']:
    if shard['source_sha256']!=sample.source_sha256:continue
    path=(Path(contract['training_manifest']).parent/shard['path']).resolve()
    assert file_sha256(path)==shard['sha256']
    data=dict(np.load(path,allow_pickle=False)); t=np.array(shard['frame_ids'])[:,None];q=np.array(shard['query_ids'])[None,:]
    assert not seen[t,q].any();seen[t,q]=True
    np.testing.assert_array_equal(data['target_visible'],sample.target_visible[t,q])
    np.testing.assert_allclose(data['target_xy'],sample.target_xy[t,q],equal_nan=True)
    use=data['target_valid'].astype(bool)
    np.testing.assert_allclose(data['candidates'][...,0,:][use],a['candidates'][t,q][...,0,:][use],atol=1e-5,rtol=0)
    for key in old:
        old[key].append(data[key][use]);new[key].append(a[mapping[key]][t,q][use])
    targets.append(data['target_xy'][use]);visibility.append(data['target_visible'][use]);saved.append(a['xy'][t,q][use]);used.append(shard['sha256'])
np.testing.assert_array_equal(seen,a['processed'])
old={k:np.concatenate(v) for k,v in old.items()};new={k:np.concatenate(v) for k,v in new.items()}
gt=np.concatenate(targets);vis=np.concatenate(visibility).astype(bool);saved=np.concatenate(saved)
stats=[];normalized=[];outputs=[]
for label,data in [('bootstrap_cache',old),('learned_rollout',new)]:
    xy=[];weights=[];prob=[]
    with torch.inference_mode():
        for start in range(0,len(gt),2048):
            d={k:torch.as_tensor(v[start:start+2048]).bool() if k=='valid' else torch.as_tensor(v[start:start+2048]).float() for k,v in data.items()}
            r=model(d['features'],d['visual_logits'],d['candidates'],d['candidate_visibility'],d['valid'])
            xy.append(r['xy'].numpy());weights.append(r['weights'].numpy());prob.append(r['visibility'].numpy())
        features=canonical_features(torch.as_tensor(data['features']).float(),torch.as_tensor(data['valid']).bool())
        normalized.append(((features-model.feature_mean)/model.feature_std.clamp_min(1e-5)).numpy())
    xy=np.concatenate(xy);weight=np.concatenate(weights);prob=np.concatenate(prob);outputs.append(xy)
    scale=np.array([256/sample.rgb.shape[2],256/sample.rgb.shape[1]])
    distance=np.linalg.norm((xy[vis]-gt[vis])*scale,axis=-1)
    stats.append(dict(input=label,visible_mean_error_px=float(np.linalg.norm(xy[vis]-gt[vis],axis=-1).mean()),
        position_threshold_accuracy=float(np.mean([np.mean(distance<t) for t in [1,2,4,8,16]])),
        visibility_accuracy=float(np.mean((prob>=.5)==vis)),mean_geometry_weight=float(weight[:,1:3].sum(-1).mean())))
np.testing.assert_allclose(outputs[1],saved,atol=.002,rtol=1e-6)
common=old['valid'].astype(bool)&new['valid'].astype(bool)
features=[]
for j,name in enumerate(FEATURE_NAMES):
    left=normalized[0][...,j][common];right=normalized[1][...,j][common]
    features.append(dict(name=name,common_valid_candidates=len(left),
        bootstrap_abs_normalized_p95=float(np.percentile(np.abs(left),95)),
        learned_abs_normalized_p95=float(np.percentile(np.abs(right),95)),
        paired_abs_difference_p95=float(np.percentile(np.abs(right-left),95)),
        bootstrap_over_5std_fraction=float(np.mean(np.abs(left)>5)),
        learned_over_5std_fraction=float(np.mean(np.abs(right)>5))))
report=dict(state='complete',video=sample.name,labeled_point_frames=len(gt),cache_shards=len(used),cache_shard_sha256=used,
    tracks_sha256=binding['tracks_sha256'],checkpoint_sha256=binding['checkpoint_sha256'],
    candidate_validity_changed_fraction=float(np.mean(old['valid']!=new['valid'])),
    same_head_on_each_input=stats,feature_distribution=features,
    scope='First complete TRAIN video, same head and RGB/queries. Cache feedback is bootstrap-generated, recording feedback learned. Independent geometry timing and window selection confound causal attribution. No held-out or training gain claim.')
out=exp/'distribution_audit';out.mkdir(exist_ok=False);(out/'audit.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items() if k not in ['cache_shard_sha256','feature_distribution']},indent=2))
