"""Merge completed caches confined to one NEW reproduction run directory.

The path guard prevents accidentally linking old experiment shards. Execution
logs from actual cache generation are still required to prove a fresh rebuild.
"""
from pathlib import Path
import argparse,json,hashlib

def digest(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
 return h.hexdigest()

def merge(inputs,run_root,output):
 root=Path(run_root).resolve(strict=True);output=Path(output).resolve()
 if not output.is_relative_to(root):raise ValueError('Output must be inside the new reproduction root')
 if output.exists():raise FileExistsError(output)
 manifests=[];source={};policies=[];shards=[];seen=set()
 for raw in inputs:
  path=Path(raw).resolve(strict=True)
  if not path.is_relative_to(root):raise ValueError('Old/external cache manifest forbidden: '+str(path))
  m=json.loads(path.read_text())
  if m.get('schema_version')!=1 or not m.get('shards') or m.get('geometry_candidate_count',0)<=0:raise ValueError('Incomplete geometry cache')
  status=path.parent/'status.json'
  if not status.exists():status=path.parent.parent/'status.json'
  if json.loads(status.read_text())['state']!='complete':raise ValueError('Cache generation not complete')
  for key in ['feature_names','baseline_provenance','geometry_intrinsics_protocol','evidence_age_protocol']:
   if manifests and m[key]!=manifests[0][key]:raise ValueError('Incompatible cache '+key)
  for entry in m['shards']:
   p=(path.parent/entry['path']).resolve(strict=True)
   if not p.is_relative_to(root):raise ValueError('Old/external shard forbidden: '+str(p))
   if p in seen:raise ValueError('Duplicate shard path')
   if entry['split']!='train' or digest(p)!=entry['sha256']:raise ValueError('Invalid TRAIN shard')
   seen.add(p);shards.append({**entry,'path':str(p)})
  manifests.append(m);source[str(path)]=digest(path);policies.append(dict(source=str(path),rollout_policy=m['rollout_policy'],query_protocol=m.get('query_protocol'),max_queries_per_cohort=m.get('max_queries_per_cohort')))
 if not manifests:raise ValueError('At least one completed cache required')
 first=manifests[0]
 result=dict(schema_version=1,feature_names=first['feature_names'],baseline_provenance=first['baseline_provenance'],geometry_intrinsics_protocol=first['geometry_intrinsics_protocol'],evidence_age_protocol=first['evidence_age_protocol'],shards=shards,source_cache_manifests=source,source_rollout_protocols=policies,rollout_policy='mixed source protocols explicitly listed',scene_count=sum(m['scene_count'] for m in manifests),geometry_candidate_count=sum(m['geometry_candidate_count'] for m in manifests),dataset_manifest_sha256=None,runtime_config_sha256=None,scope='Reproduction cache manifests confined to this run; grouped variant is augmentation of the same training scene, not independent data')
 output.parent.mkdir(parents=True,exist_ok=True)
 with output.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
 return result

if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',action='append',required=True);p.add_argument('--run-root',required=True);p.add_argument('--output',required=True);a=p.parse_args();r=merge(a.input,a.run_root,a.output);print(json.dumps(dict(shards=len(r['shards']),geometry_candidates=r['geometry_candidate_count'])))
