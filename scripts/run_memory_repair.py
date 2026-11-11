"""Run the accepted verified historical geometry-memory model."""
from pathlib import Path
import argparse,hashlib,json,subprocess
R=Path(__file__).resolve().parents[1];O=R/'runs/memory-repair-415'
def verify():
 f=O/'frozen_delivery.json';record=json.loads(f.read_text())
 if record['state']!='accepted_frozen':raise ValueError('Delivery is not accepted')
 for name,digest in record['bindings'].items():
  if hashlib.sha256((R/name).read_bytes()).hexdigest()!=digest:raise ValueError('Frozen artifact changed: '+name)
 return record

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path);p.add_argument('--dataset-manifest',type=Path,default=R/'runs/robot-development-expanded-002/manifest.json');p.add_argument('--split',default='dev');p.add_argument('--indices',default='0,1,2,3,4,5,6,7,8,9,10,11');p.add_argument('--verify-only',action='store_true');a=p.parse_args();record=verify();print('Frozen delivery verified:',record['state'],record['overall21_changes_pp'],flush=True)
 if a.verify_only:return
 if a.output is None:p.error('--output is required')
 if a.output.exists():p.error('--output must be a new directory')
 if not a.dataset_manifest.is_file():p.error('Dataset manifest not found')
 subprocess.run([str(R/'.envs/trackon312/bin/python'),'-u',str(O/'evaluate_resource.py'),'--checkpoint',str(R/record['primary_checkpoint']),'--output',str(a.output.resolve()),'--manifest',str(a.dataset_manifest.resolve()),'--split',a.split,'--indices',a.indices],cwd=R,check=True)
if __name__=='__main__':main()
