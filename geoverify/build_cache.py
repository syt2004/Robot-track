"""Produce training-only candidate caches from complete Kubric scene manifests."""
import argparse
import json
from pathlib import Path
import numpy as np
from .cache import validate_arrays
from .datasets import file_sha256
from .feature_schema import FEATURE_NAMES
from .geometry_worker import atomic_json
from .kubric import iter_kubric
from .video_runner import run_query_cohorts,save_video_predictions


class SceneCacheWriter:
    def __init__(self,sample,output,scene_index,batch_frames=32):
        if sample.split!='train' or batch_frames<1:
            raise ValueError('Cache writer requires train scenes and positive batch size')
        self.sample,self.output,self.scene_index=sample,Path(output),scene_index
        self.batch_frames=batch_frames
        self.ids=None
        self.rows=[]
        self.frames=[]
        self.shards=[]
        self.geometry_candidates=0
        self.seen=np.zeros(sample.target_visible.shape,bool)

    def observe(self,frame,ids,output):
        ids=np.asarray(ids)
        if output.get('source')!='training_bootstrap':
            raise ValueError('Initial cache must identify its bootstrap rollout policy')
        if ids.dtype.kind not in 'iu' or not len(ids) or len(np.unique(ids))!=len(ids) or ((ids<0)|(ids>=self.seen.shape[1])).any():
            raise ValueError('Invalid cache query IDs')
        if not 0<=frame<len(self.seen) or (frame<self.sample.query_txy[ids,0]).any() or self.seen[frame,ids].any():
            raise ValueError('Duplicate, premature or invalid cache observation')
        if self.ids is not None and not np.array_equal(ids,self.ids):
            self.flush()
        self.ids=ids.copy()
        self.seen[frame,ids]=True
        row={key:np.array(output[key],copy=True) for key in
             ('features','visual_logits','candidates','candidate_visibility','valid')}
        # Labels are joined only after the model emitted its predictions.
        row.update(target_xy=self.sample.target_xy[frame,ids].copy(),
                   target_visible=self.sample.target_visible[frame,ids].copy(),
                   target_valid=frame>self.sample.query_txy[ids,0])
        self.rows.append(row)
        self.frames.append(frame)
        if len(self.rows)>=self.batch_frames:
            self.flush()

    def flush(self):
        if not self.rows:
            return
        arrays={key:np.stack([row[key] for row in self.rows]) for key in self.rows[0]}
        arrays=validate_arrays(arrays,len(FEATURE_NAMES))
        self.geometry_candidates+=int(arrays['valid'][...,1:3].sum())
        relative='scene_%06d/chunk_%06d.npz'%(self.scene_index,len(self.shards))
        save_video_predictions(self.output/relative,arrays)
        self.shards.append(dict(path=relative,sha256=file_sha256(self.output/relative),split='train',
                                width=self.sample.rgb.shape[2],height=self.sample.rgb.shape[1],
                                scene=self.sample.name,source_sha256=self.sample.source_sha256,
                                frame_ids=list(self.frames),query_ids=self.ids.tolist()))
        self.rows=[]
        self.frames=[]

    def finish(self):
        self.flush()
        expected=np.arange(len(self.seen))[:,None]>=self.sample.query_txy[:,0]
        if not np.array_equal(self.seen,expected):
            raise ValueError('Incomplete full-scene training observations')
        return self.shards


def build_cache(dataset_manifest,runtime_config,output,batch_frames=32,max_queries_per_cohort=128):
    from .runtime import RuntimeFactory
    root=Path(output).resolve()
    root.mkdir(parents=True,exist_ok=False)
    stage='input_validation'
    atomic_json(root/'status.json',dict(state='running',stage=stage))
    try:
        if batch_frames<1:
            raise ValueError('batch-frames must be positive')
        if isinstance(max_queries_per_cohort,bool) or not isinstance(max_queries_per_cohort,int) or max_queries_per_cohort<1:
            raise ValueError('Query cohort limit must be a positive integer')
        manifest=dict(schema_version=1,feature_names=list(FEATURE_NAMES),shards=[],
                      rollout_policy='verifier_identity_residual_bootstrap',
                      dataset_manifest_sha256=file_sha256(dataset_manifest),
                      runtime_config_sha256=file_sha256(runtime_config),
                      geometry_candidate_count=0,scene_count=0,
                      max_queries_per_cohort=max_queries_per_cohort,
                      query_protocol='all queries, stable source order, bounded independent start-frame cohorts; not joint-query equivalent')
        (root/'runtime_config.snapshot.json').write_bytes(Path(runtime_config).read_bytes())
        (root/'dataset_manifest.snapshot.json').write_bytes(Path(dataset_manifest).read_bytes())
        stage='model_loading'
        atomic_json(root/'status.json',dict(state='running',stage=stage))
        factory=RuntimeFactory(runtime_config,purpose='cache_bootstrap')
        manifest['geometry_intrinsics_protocol']=json.loads(Path(runtime_config).read_text()).get('geometry_intrinsics_protocol')
        manifest['baseline_provenance']=factory.provenance
        stage='cache_generation'
        atomic_json(root/'status.json',dict(state='running',stage=stage))
        for index,sample in enumerate(iter_kubric(dataset_manifest,'train')):
            print('Cache scene %d: %s'%(index+1,sample.name),flush=True)
            writer=SceneCacheWriter(sample,root,index,batch_frames)
            run_query_cohorts(sample.rgb,sample.timestamps,sample.query_txy,factory,on_output=writer.observe,
                              max_queries_per_cohort=max_queries_per_cohort)
            manifest['shards'].extend(writer.finish())
            manifest['geometry_candidate_count']+=writer.geometry_candidates
            manifest['scene_count']+=1
            atomic_json(root/'progress.json',manifest)
        if not manifest['shards'] or manifest['geometry_candidate_count']==0:
            raise ValueError('No complete training cache with actual geometry candidates')
        atomic_json(root/'manifest.json',manifest)
        atomic_json(root/'status.json',dict(state='complete',stage='finished',scene_count=manifest['scene_count']))
    except BaseException as error:
        atomic_json(root/'status.json',dict(state='failed',stage=stage,error=type(error).__name__+': '+str(error)))
        raise
    return root/'manifest.json'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-manifest',required=True)
    parser.add_argument('--runtime-config',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--batch-frames',type=int,default=32)
    parser.add_argument('--max-queries-per-cohort',type=int,default=128,
                        help='Bound model query memory; retain every query and all frames, replaying independently')
    args=parser.parse_args()
    if args.batch_frames<1:
        parser.error('batch-frames must be positive')
    print(build_cache(args.dataset_manifest,args.runtime_config,args.output,args.batch_frames,args.max_queries_per_cohort))


if __name__=='__main__':
    main()
