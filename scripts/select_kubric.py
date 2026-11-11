"""Deterministic, explicitly authorized small Kubric training/validation subset."""
import argparse
import hashlib
import json
from pathlib import Path


def select(inventory,count=2,validation_count=1,seed=20261029,prior=None):
    files=inventory['files']
    if len(files)!=inventory['archive_count'] or len({f['path'] for f in files})!=len(files):
        raise ValueError('Incomplete or duplicate source inventory')
    if not 0<validation_count<count<=len(files):
        raise ValueError('Require nonempty train/validation subsets within inventory size')
    def rank(entry):
        return hashlib.sha256((str(seed)+':'+entry['path']).encode()).hexdigest(),entry['path']
    ordered=sorted(files,key=rank)
    entries=[]
    if prior is not None:
        if any(prior.get(k)!=inventory.get(k) for k in ('repository','revision')) or prior.get('seed')!=seed:
            raise ValueError('Prior selection source/seed mismatch')
        by_path={entry['path']:entry for entry in files}
        seen=set()
        for entry in prior['files']:
            path=entry['path']
            if path in seen or path not in by_path or entry['split'] not in ('train','val'):
                raise ValueError('Invalid prior selection')
            if any(entry.get(k)!=v for k,v in by_path[path].items()):
                raise ValueError('Prior archive identity mismatch')
            seen.add(path)
            entries.append(dict(entry))
    else:
        seen=set()
    required={'val':validation_count,'train':count-validation_count}
    for split in required:
        required[split]-=sum(e['split']==split for e in entries)
        if required[split]<0:
            raise ValueError('Expansion cannot remove or reassign prior scenes')
    available=iter(e for e in ordered if e['path'] not in seen)
    for split in ('val','train'):
        for _ in range(required[split]):
            entries.append(dict(next(available),split=split))
    return dict(schema_version=1,repository=inventory['repository'],revision=inventory['revision'],
                selection='Preserve prior splits; add sha256(seed:archive_path) ranked unused scenes; independent of model outputs',seed=seed,
                source_archive_count=len(files),selected_archive_count=count,
                train_archive_count=count-validation_count,validation_archive_count=validation_count,
                total_compressed_bytes=sum(f['bytes'] for f in entries),
                scope='user-authorized subset; not the complete published Kubric dataset',
                files=sorted(entries,key=lambda f:f['path']))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inventory',default='configs/kubric.archives.lock.json')
    parser.add_argument('--output',default='configs/kubric.small.lock.json')
    parser.add_argument('--count',type=int,default=2)
    parser.add_argument('--validation-count',type=int,default=1)
    parser.add_argument('--seed',type=int,default=20261029)
    parser.add_argument('--prior-selection',help='Preserve all existing scenes and splits when expanding')
    args=parser.parse_args()
    source=Path(args.inventory).read_bytes()
    prior_bytes=Path(args.prior_selection).read_bytes() if args.prior_selection else None
    result=select(json.loads(source),args.count,args.validation_count,args.seed,
                  json.loads(prior_bytes) if prior_bytes else None)
    if prior_bytes:
        result['prior_selection_sha256']=hashlib.sha256(prior_bytes).hexdigest()
    result['source_inventory_sha256']=hashlib.sha256(source).hexdigest()
    output=Path(args.output)
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x') as stream:
        json.dump(result,stream,indent=2)
        stream.write('\n')
    print(json.dumps({k:v for k,v in result.items() if k!='files'},indent=2))


if __name__=='__main__':
    main()
