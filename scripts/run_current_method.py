"""Run the current v5 candidate from verified source, never cwd geoverify.

Example: .envs/trackon312/bin/python scripts/run_current_method.py train --help
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1]

def configuration(root=ROOT):
 root=Path(root).resolve();path=root/'configs/current-method-candidate.json'
 config=json.loads(path.read_text())
 for key,hashkey in [('source_manifest','source_manifest_sha256'),('runner','runner_sha256')]:
  p=(root/config[key]).resolve(strict=True)
  if not p.is_relative_to(root):raise ValueError('Candidate path outside project: '+key)
  if hashlib.sha256(p.read_bytes()).hexdigest()!=config[hashkey]:raise ValueError('Candidate SHA256 mismatch: '+key)
 source=(root/config['source']).resolve(strict=True)
 if not source.is_relative_to(root):raise ValueError('Candidate source outside project')
 for name,digest in json.loads((root/config['source_manifest']).read_text()).items():
  p=(source/name).resolve(strict=True)
  if not p.is_relative_to(source) or hashlib.sha256(p.read_bytes()).hexdigest()!=digest:raise ValueError('Candidate source SHA256 mismatch: '+name)
 return config

def main():
 parser=argparse.ArgumentParser(description=__doc__)
 parser.add_argument('--describe',action='store_true')
 parser.add_argument('command',nargs='?',choices=['train','cache','promote','evaluate','demo'])
 parser.add_argument('arguments',nargs=argparse.REMAINDER)
 args=parser.parse_args();config=configuration()
 if args.describe:print(json.dumps(config,indent=2));return
 if not args.command:parser.error('Choose a command or --describe')
 command=[sys.executable,str(ROOT/config['runner']),'--source',str(ROOT/config['source']),'--manifest',str(ROOT/config['source_manifest']),config['commands'][args.command],*args.arguments]
 # Preserve caller cwd and explicit runtime/data paths. Geometry workers locate
 # the verified package from __file__, so they use the same copied source too.
 raise SystemExit(subprocess.call(command))

if __name__=='__main__':main()
