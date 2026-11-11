"""Create an explicit reproducible development subset without changing raw data.

Select by seeded source index among queries first visible on frame zero. This
keeps complete videos and removes repeated model startups for late-query cohorts.
It is not the full-query benchmark protocol or evidence of dataset-wide quality.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from .datasets import file_sha256
from .kubric import iter_kubric


def select(source,output,count=256,seed=20261030):
    source,output=Path(source).resolve(),Path(output).resolve()
    if count<1 or output.exists():
        raise ValueError('Positive count and new output path required')
    manifest=json.loads(source.read_text())
    if manifest.get('query_selection') or any('query_indices' in s for s in manifest['scenes']):
        raise ValueError('Select from a full-query source manifest only')
    counts={}
    indices={}
    for split in sorted({s['split'] for s in manifest['scenes']}):
        for sample in iter_kubric(source,split):
            eligible=np.flatnonzero(sample.query_txy[:,0]==0)
            rank=sorted(eligible,key=lambda i:hashlib.sha256(('%d:%s:%d'%(seed,sample.name,i)).encode()).digest())
            indices[sample.name]=sorted(int(i) for i in rank[:count])
            counts[sample.name]=dict(source_queries=len(sample.query_txy),eligible_first_frame_queries=len(eligible),selected_queries=len(indices[sample.name]))
            if not indices[sample.name]:
                raise ValueError('No eligible first-frame query: '+sample.name)
    for scene in manifest['scenes']:
        scene['query_indices']=indices[scene['name']]
        for entry in [scene['annotation'],*scene['frames']]:
            entry['path']=str((source.parent/entry['path']).resolve())
    manifest['query_selection']=dict(source_manifest=str(source),source_manifest_sha256=file_sha256(source),
        seed=seed,max_queries_per_scene=count,counts=counts,
        protocol='lowest sha256(seed:scene:source_query_index) among frame-zero first-visible queries; indices sorted in source order',
        scope='explicit development point subset; all RGB frames retained; not full-query benchmark')
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x') as f: json.dump(manifest,f,indent=2)
    return output


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',required=True); p.add_argument('--output',required=True)
    p.add_argument('--count',type=int,default=256); p.add_argument('--seed',type=int,default=20261030)
    a=p.parse_args(); print(select(a.source,a.output,a.count,a.seed))
