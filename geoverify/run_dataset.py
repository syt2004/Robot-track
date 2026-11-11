"""Run every video/query in one locked RoboTAP split and score saved outputs."""
import argparse
import json
from pathlib import Path
import time
from .datasets import iter_evaluation,file_sha256
from .evaluate import evaluate
from .geometry_worker import atomic_json
from .video_runner import run_query_cohorts,save_video_predictions,fusion_usage
from .overlay import render_overlay,render_comparison,rigid_event_markers


def run_dataset(dataset_manifest,split,output,factory,runtime_config,visibility_threshold=.5,
                overlays=True,runner=run_query_cohorts):
    if getattr(factory,'purpose','inference')!='inference':
        raise ValueError('Training bootstrap cannot be evaluated as the complete model')
    training_sources=getattr(getattr(factory,'calibrator',None),'training_source_sha256',None)
    if not training_sources:
        raise ValueError('Held-out evaluation requires checkpoint training source identities')
    output=Path(output).resolve()
    output.mkdir(parents=True,exist_ok=False)
    dataset_manifest=Path(dataset_manifest).resolve()
    runtime_config=Path(runtime_config).resolve()
    manifest=dict(schema_version=1,protocol='tapvid_first',split=split,
                  dataset_manifest_sha256=file_sha256(dataset_manifest),
                  baseline_provenance=factory.provenance,videos=[],
                  run_metadata=dict(runtime_config_sha256=file_sha256(runtime_config),
                                    calibrator_sha256=factory.config['calibrator_sha256'],
                                    replay_protocol='capture_paced_start_frame_cohorts',
                                    visibility_threshold=visibility_threshold,
                                    training_source_sha256=list(training_sources),
                                    training_overlap_check='source_annotation_sha256',
                                    object_grouping='none',source_dataset_manifest=str(dataset_manifest)))
    (output/'runtime_config.snapshot.json').write_bytes(runtime_config.read_bytes())
    (output/'dataset_manifest.snapshot.json').write_bytes(dataset_manifest.read_bytes())
    started=time.monotonic()
    atomic_json(output/'status.json',dict(state='running',completed_videos=0))
    try:
        for index,sample in enumerate(iter_evaluation(dataset_manifest,split)):
            print('Video %d: %s, %d frames, %d queries'%(index+1,sample.name,len(sample.rgb),len(sample.query_txy)),flush=True)
            if sample.source_sha256 in training_sources:
                raise ValueError('Evaluation source overlaps calibrator training data: '+sample.name)
            # No labels or inferred GT object identities cross the runner boundary.
            arrays=runner(sample.rgb,sample.timestamps,sample.query_txy,factory,
                          visibility_threshold=visibility_threshold,record_candidates=True)
            if str(arrays['tracker_source'])!=factory.provenance['track_on']['variant']:
                raise ValueError('Output tracker differs from configured baseline')
            relative='video_%06d/tracks.npz'%index
            save_video_predictions(output/relative,arrays)
            entry=dict(name=sample.name,path=relative,sha256=file_sha256(output/relative),
                       frames=len(sample.rgb),queries=len(sample.query_txy))
            if 'fusion_weights_recorded' in arrays:
                entry['fusion_usage']=fusion_usage(arrays)
            motion_path=(output/relative).with_name('object_motion.json')
            atomic_json(motion_path,dict(schema_version=1,
                        enabled=factory.config.get('pipeline',{}).get('rigid_threshold') is not None,
                        coordinate_convention='independent_window_world_gauges_not_robot_base',
                        events=json.loads(str(arrays.get('object_motion_json','[]')))))
            entry['object_motion']=str(motion_path.relative_to(output))
            entry['object_motion_sha256']=file_sha256(motion_path)
            if overlays:
                overlay_path=(output/relative).with_name('overlay.mp4')
                render_overlay(sample.rgb,sample.timestamps,arrays['xy'],arrays['visible'],
                               arrays['confidence'],overlay_path,
                               'GeoVerify | '+str(arrays['tracker_source']),
                               rigid_inliers=rigid_event_markers(arrays))
                entry['overlay']=str(overlay_path.relative_to(output))
                entry['overlay_sha256']=file_sha256(overlay_path)
                comparison_path=overlay_path.with_name('comparison.mp4')
                render_comparison(sample.rgb,sample.timestamps,arrays,comparison_path)
                entry['comparison']=str(comparison_path.relative_to(output))
                entry['comparison_sha256']=file_sha256(comparison_path)
                from .diagnostics import render_diagnostics
                diagnostic_root=overlay_path.parent/'diagnostics'
                report=render_diagnostics(sample,arrays,diagnostic_root)
                entry['diagnostics']=str((diagnostic_root/'diagnostics.json').relative_to(output))
                entry['diagnostics_sha256']=file_sha256(diagnostic_root/'diagnostics.json')
            manifest['videos'].append(entry)
            atomic_json(output/'progress.json',manifest)
            atomic_json(output/'status.json',dict(state='running',completed_videos=len(manifest['videos'])))
        if not manifest['videos']:
            raise ValueError('Dataset split is empty')
        atomic_json(output/'predictions.json',manifest)
        result=evaluate(dataset_manifest,output/'predictions.json',split)
        atomic_json(output/'metrics.json',result)
        atomic_json(output/'status.json',dict(state='complete',completed_videos=len(manifest['videos']),
                                             wall_seconds=time.monotonic()-started))
        return result
    except BaseException as error:
        atomic_json(output/'status.json',dict(state='failed',completed_videos=len(manifest['videos']),
                                             error=type(error).__name__+': '+str(error)))
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-manifest',required=True)
    parser.add_argument('--split',required=True)
    parser.add_argument('--runtime-config',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--visibility-threshold',type=float,default=.5)
    parser.add_argument('--no-overlays',action='store_true',help='Numerical evaluation only; not final visual acceptance')
    args=parser.parse_args()
    from .runtime import RuntimeFactory
    factory=RuntimeFactory(args.runtime_config)
    run_dataset(args.dataset_manifest,args.split,args.output,factory,args.runtime_config,
                args.visibility_threshold,not args.no_overlays)


if __name__=='__main__':
    main()
