"""Rebuild grouped TRAIN cache from source RGB and first-frame instance IDs.

Explicit frozen source directory; never reuse an existing geometry cache. With
--validate-only, validate inputs and exit before loading any models.
"""
from pathlib import Path
import argparse,sys,json,hashlib
import numpy as np
p=argparse.ArgumentParser(description=__doc__)
for name in ['source','dataset-manifest','groups','runtime-config','output']:p.add_argument('--'+name,required=True)
p.add_argument('--validate-only',action='store_true');args=p.parse_args()
src=Path(args.source).resolve();sys.path.insert(0,str(src))
from geoverify.datasets import file_sha256
from geoverify.kubric import iter_kubric,read_kubric_annotations
from geoverify.geometry_worker import atomic_json
out=Path(args.output).resolve();out.mkdir(parents=True,exist_ok=False)
status=dict(state='running',stage='input_validation');atomic_json(out/'status.json',status)
try:
 for name,h in json.loads((src.parent/'source_manifest.json').read_text()).items():assert file_sha256(src/name)==h,name
 dataset=Path(args.dataset_manifest).resolve();config=Path(args.runtime_config).resolve();groups=Path(args.groups).resolve();g=json.loads(groups.read_text());c=json.loads(config.read_text());samples=list(iter_kubric(dataset,'train'));assert len(samples)==1;s=samples[0]
 assert c['purpose']=='cache_bootstrap';assert c['pipeline']['rigid_threshold']>0 and c['pipeline']['rigid_reprojection_threshold']>0
 assert s.name==g['scene'] and (s.query_txy[:,0]==0).all() and g['initial_frame']==0
 assert hashlib.sha256(s.query_txy.tobytes()).hexdigest()==g['query_txy_sha256'];assert s.source_sha256==g['source_annotation_sha256']
 assert file_sha256(groups.parent/'first_frame_instances.png')==g['first_mask_sha256']
 ids=np.asarray(g['object_ids'],np.int64);assert ids.shape==(len(s.query_txy),) and (ids>=-1).all()
 from PIL import Image
 mask=np.array(Image.open(groups.parent/'first_frame_instances.png'));assert mask.shape==s.rgb.shape[1:3]
 annotation=json.loads(dataset.read_text())['scenes'][0]['annotation']['path']
 original=read_kubric_annotations((dataset.parent/annotation).resolve())['segmentations'][0,...,0]
 np.testing.assert_array_equal(mask,original)
 derived=[]
 for _,x,y in s.query_txy:
  x,y=int(np.rint(x)),int(np.rint(y));patch=mask[max(0,y-1):y+2,max(0,x-1):x+2];label=int(mask[y,x])
  derived.append(label if label>0 and patch.shape==(3,3) and np.all(patch==label) else -1)
 np.testing.assert_array_equal(ids,np.asarray(derived),err_msg='Groups must derive from first-frame masks only')
 inputs=dict(source_manifest_sha256=file_sha256(src.parent/'source_manifest.json'),dataset_manifest_sha256=file_sha256(dataset),runtime_config_sha256=file_sha256(config),initialization_groups_sha256=file_sha256(groups),frames=len(s.rgb),queries=len(s.query_txy),scene=s.name,source_sha256=s.source_sha256)
 atomic_json(out/'input_audit.json',inputs)
 if args.validate_only:
  status.update(state='validated_only',stage='finished',note='No model loaded, no cache generated, not a frozen or reproduced experiment');atomic_json(out/'status.json',status);print(json.dumps(inputs,indent=2));sys.exit(0)
 from geoverify.runtime import RuntimeFactory
 from geoverify.build_cache import SceneCacheWriter
 from geoverify.video_runner import run_query_cohorts,save_video_predictions
 from geoverify.feature_schema import FEATURE_NAMES
 cache=out/'cache';cache.mkdir();status['stage']='model_loading';atomic_json(out/'status.json',status);factory=RuntimeFactory(config,purpose='cache_bootstrap')
 writer=SceneCacheWriter(s,cache,0);status['stage']='cache_generation';atomic_json(out/'status.json',status)
 arrays=run_query_cohorts(s.rgb,s.timestamps,s.query_txy,factory,object_ids=ids,on_output=writer.observe,record_candidates=True)
 shards=writer.finish();assert writer.geometry_candidates>0;save_video_predictions(cache/'rollout.npz',arrays)
 (cache/'dataset_manifest.snapshot.json').write_bytes(dataset.read_bytes());(cache/'runtime_config.snapshot.json').write_bytes(config.read_bytes())
 manifest=dict(schema_version=1,feature_names=list(FEATURE_NAMES),shards=shards,baseline_provenance=factory.provenance,geometry_intrinsics_protocol=c['geometry_intrinsics_protocol'],evidence_age_protocol=c.get('evidence_age_protocol'),rollout_policy='verifier_identity_residual_bootstrap_with_first_frame_instance_groups_and_optional_rigid',dataset_manifest_sha256=inputs['dataset_manifest_sha256'],runtime_config_sha256=inputs['runtime_config_sha256'],scene_count=1,geometry_candidate_count=writer.geometry_candidates,initialization_groups_sha256=inputs['initialization_groups_sha256'],query_protocol='all original queries jointly; no cohort subdivision of grouped objects',rollout_sha256=file_sha256(cache/'rollout.npz'),scope='TRAIN input variant of existing Kubric scene; not independent data or robot validation')
 atomic_json(cache/'manifest.json',manifest);status.update(state='complete',stage='finished',geometry_candidates=writer.geometry_candidates);atomic_json(out/'status.json',status)
except SystemExit:raise
except BaseException as e:
 status.update(state='failed',error=repr(e));atomic_json(out/'status.json',status);raise
