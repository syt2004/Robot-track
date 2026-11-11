"""Verify selected archives and prepare full RGB/annotation scene manifests.

Only the unused depth/render assets are omitted, as allowed by the upstream
dataset instructions. Every RGB frame and the complete trajectory annotation
are retained. Partial archives are never consumed as training data.
"""
import argparse
import json
from pathlib import Path,PurePosixPath
import shutil
import tarfile
from .datasets import file_sha256
from .geometry_worker import atomic_json


def selected_members(archive,scene):
    chosen=[]
    names=set()
    for member in archive.getmembers():
        path=PurePosixPath(member.name)
        if path.is_absolute() or '..' in path.parts or not path.parts or path.parts[0]!=scene:
            raise ValueError('Archive member escapes expected scene')
        if member.issym() or member.islnk() or not (member.isfile() or member.isdir()):
            raise ValueError('Archive links/special files are not allowed')
        normalized=str(path)
        if normalized in names:
            raise ValueError('Duplicate archive member')
        names.add(normalized)
        if member.isdir():
            continue
        annotation=normalized==scene+'/'+scene+'.npy'
        frame=len(path.parts)==3 and path.parts[1]=='frames' and path.suffix.lower() in ('.png','.jpg','.jpeg')
        if annotation or frame:
            chosen.append(member)
    if not any(PurePosixPath(m.name).suffix=='.npy' for m in chosen):
        raise ValueError('Missing scene annotations')
    if not any(PurePosixPath(m.name).parent.name=='frames' for m in chosen):
        raise ValueError('Missing scene RGB frames')
    return chosen


def prepare(selection_path,archive_root,output,fps,fps_description):
    import math
    if not math.isfinite(fps) or fps<=0 or not fps_description.strip():
        raise ValueError('Explicit positive FPS and its source/assumption required')
    selection_path=Path(selection_path).resolve()
    selection=json.loads(selection_path.read_text())
    entries=selection['files']
    if len(entries)!=selection['selected_archive_count'] or len({e['path'] for e in entries})!=len(entries):
        raise ValueError('Incomplete or duplicate selected archive list')
    archive_root=Path(archive_root).resolve()
    verified=[]
    for entry in entries:
        relative=PurePosixPath(entry['path'])
        if len(relative.parts)!=1 or not relative.name.endswith('.tar.gz') or entry['split'] not in ('train','val'):
            raise ValueError('Invalid archive filename/split')
        path=archive_root/relative.name
        if not path.is_file() or path.stat().st_size!=entry['bytes'] or file_sha256(path)!=entry['sha256']:
            raise ValueError('Missing/incomplete/unverified archive: '+str(path))
        verified.append((entry,path))
    output=Path(output).resolve()
    output.mkdir(parents=True,exist_ok=False)
    manifest=dict(schema_version=1,dataset='cotracker3_kubric',scenes=[],
                  selection_manifest_sha256=file_sha256(selection_path),
                  source_revision=selection['revision'],fps_description=fps_description,
                  omitted_assets='non-RGB render/depth assets only')
    atomic_json(output/'status.json',dict(state='preparing'))
    try:
        for entry,path in verified:
            scene=entry['path'][:-len('.tar.gz')]
            files=[]
            with tarfile.open(path,'r:gz') as archive:
                members=selected_members(archive,scene)
                for member in members:
                    destination=output/PurePosixPath(member.name)
                    destination.parent.mkdir(parents=True,exist_ok=True)
                    with archive.extractfile(member) as source,destination.open('xb') as target:
                        shutil.copyfileobj(source,target,1024*1024)
                    if destination.stat().st_size!=member.size:
                        raise ValueError('Incomplete extracted archive member')
                    files.append(dict(path=str(destination.relative_to(output)),sha256=file_sha256(destination)))
            annotation=next(f for f in files if f['path']==scene+'/'+scene+'.npy')
            frames=sorted([f for f in files if PurePosixPath(f['path']).parent.name=='frames'],key=lambda f:f['path'])
            manifest['scenes'].append(dict(name=scene,split=entry['split'],fps=fps,
                                           source_archive_sha256=entry['sha256'],annotation=annotation,frames=frames))
            atomic_json(output/'progress.json',manifest)
        # Parse every retained annotation/frame and verify full temporal lengths
        # before advertising a ready dataset to the cache producer.
        atomic_json(output/'manifest.pending.json',manifest)
        from .kubric import iter_kubric
        for split in sorted({e['split'] for e in entries}):
            for sample in iter_kubric(output/'manifest.pending.json',split):
                print('Validated %s %s: %d frames, %d queries'%(split,sample.name,len(sample.rgb),len(sample.query_txy)),flush=True)
        (output/'manifest.pending.json').replace(output/'manifest.json')
        atomic_json(output/'status.json',dict(state='complete',scenes=len(manifest['scenes'])))
    except BaseException as error:
        atomic_json(output/'status.json',dict(state='failed',error=type(error).__name__+': '+str(error)))
        raise
    return output/'manifest.json'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--selection',default='configs/kubric.small.lock.json')
    parser.add_argument('--archive-root',default='data/downloads/kubric')
    parser.add_argument('--output',required=True)
    parser.add_argument('--fps',type=float,required=True)
    parser.add_argument('--fps-description',required=True)
    args=parser.parse_args()
    print(prepare(args.selection,args.archive_root,args.output,args.fps,args.fps_description))


if __name__=='__main__':
    main()
