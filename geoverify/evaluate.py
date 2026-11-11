"""Score complete saved predictions against a checksum-locked RoboTAP split.

This is a scoring entry point, not an inference runner. Prediction NPZ files
must contain query_txy, timestamps, xy and visible; no pickled objects are read.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from .cache import validate_provenance
from .datasets import file_sha256, iter_evaluation
from .metrics import tapvid_first


def aggregate_video_metrics(records):
    if not records:
        raise ValueError('Cannot score an empty split')
    keys=set(next(iter(records.values())))
    if any(set(row)!=keys for row in records.values()):
        raise ValueError('Metric schema differs between videos')
    means,counts={},{}
    for key in sorted(keys):
        values=np.asarray([row[key] for row in records.values()],float)
        if np.isinf(values).any():
            raise ValueError('Infinite metric')
        valid=np.isfinite(values)
        counts[key]=int(valid.sum())
        means[key]=float(values[valid].mean()) if valid.any() else None
    return means,counts


def evaluate(dataset_manifest,prediction_manifest,split):
    prediction_manifest=Path(prediction_manifest).resolve()
    manifest=json.loads(prediction_manifest.read_text())
    if manifest.get('schema_version')!=1 or manifest.get('protocol')!='tapvid_first':
        raise ValueError('Expected prediction schema 1 / tapvid_first protocol')
    if manifest.get('dataset_manifest_sha256')!=file_sha256(dataset_manifest) or manifest.get('split')!=split:
        raise ValueError('Prediction dataset identity/split mismatch')
    validate_provenance(manifest['baseline_provenance'])
    entries={}
    paths=set()
    for entry in manifest['videos']:
        path=(prediction_manifest.parent/entry['path']).resolve()
        if entry['name'] in entries or path in paths:
            raise ValueError('Duplicate prediction video or file')
        entries[entry['name']]=(entry,path)
        paths.add(path)
    metrics={}
    raw_metrics={}
    frames=points=0
    for sample in iter_evaluation(dataset_manifest,split):
        if sample.name not in entries:
            raise ValueError('Missing full-video prediction: '+sample.name)
        entry,path=entries.pop(sample.name)
        if file_sha256(path)!=entry['sha256']:
            raise ValueError('Prediction checksum mismatch: '+sample.name)
        with np.load(path,allow_pickle=False) as arrays:
            if not np.array_equal(arrays['query_txy'],sample.query_txy):
                raise ValueError('Query identities/order differ: '+sample.name)
            if not np.array_equal(arrays['timestamps'],sample.timestamps):
                raise ValueError('Full-video timestamps differ: '+sample.name)
            visible=arrays['visible']
            if not np.isin(visible,[0,1]).all():
                raise ValueError('Predicted visibility must be binary')
            metrics[sample.name]=tapvid_first(
                sample.query_txy,sample.target_xy,sample.target_visible,
                arrays['xy'],visible,sample.rgb.shape[2],sample.rgb.shape[1])
            diagnostic_keys={'candidates','candidate_valid','candidate_visibility','visibility_threshold'}
            present=(diagnostic_keys-{'visibility_threshold'}).intersection(arrays.files)
            if present and not diagnostic_keys.issubset(arrays.files):
                raise ValueError('Incomplete raw Track-On comparison diagnostics')
            if present:
                candidates=arrays['candidates']
                valid=arrays['candidate_valid']
                probability=arrays['candidate_visibility']
                shape=sample.target_visible.shape
                if candidates.shape!=shape+(4,2) or valid.shape!=shape+(4,) or probability.shape!=valid.shape:
                    raise ValueError('Raw comparison shape mismatch')
                active=np.arange(len(sample.rgb))[:,None]>=sample.query_txy[None,:,0]
                if not np.isin(valid,[0,1]).all() or not valid[:,:,0][active].all():
                    raise ValueError('Raw Track-On must cover every active query')
                raw_probability=probability[:,:,0]
                if not np.isfinite(raw_probability[active]).all() or ((raw_probability[active]<0)|(raw_probability[active]>1)).any():
                    raise ValueError('Invalid raw Track-On visibility probability')
                threshold=float(arrays['visibility_threshold'])
                if not np.isfinite(threshold) or not 0<=threshold<=1:
                    raise ValueError('Invalid raw comparison visibility threshold')
                raw_metrics[sample.name]=tapvid_first(
                    sample.query_txy,sample.target_xy,sample.target_visible,candidates[:,:,0],
                    (raw_probability>=threshold)&active,sample.rgb.shape[2],sample.rgb.shape[1])
        frames+=len(sample.rgb)
        points+=len(sample.query_txy)
    if entries:
        raise ValueError('Predictions contain videos outside requested split')
    means,counts=aggregate_video_metrics(metrics)
    comparison=dict(available=False)
    if raw_metrics:
        if set(raw_metrics)!=set(metrics):
            raise ValueError('Raw comparison must cover every evaluated video, not a selected subset')
        raw_means,raw_counts=aggregate_video_metrics(raw_metrics)
        paired={name:{key:metrics[name][key]-raw_metrics[name][key] for key in metrics[name]}
                for name in metrics}
        deltas,delta_counts=aggregate_video_metrics(paired)
        comparison=dict(available=True,source='candidate_0_from_same_causal_replay',
                        protocol='tapvid_first',metrics=raw_means,defined_video_counts=raw_counts,
                        fused_minus_raw=deltas,paired_defined_video_counts=delta_counts,
                        per_video={name:{key:float(value) if np.isfinite(value) else None
                                         for key,value in row.items()} for name,row in raw_metrics.items()},
                        timing_note='Shared pipeline raw output; not an independent baseline throughput run')
    # JSON null explicitly denotes undefined metrics, never a fabricated zero.
    per_video={name:{key:float(value) if np.isfinite(value) else None
                     for key,value in row.items()} for name,row in metrics.items()}
    return dict(schema_version=1,protocol='tapvid_first',split=split,
                dataset_manifest_sha256=file_sha256(dataset_manifest),
                prediction_manifest_sha256=file_sha256(prediction_manifest),
                baseline_provenance=manifest['baseline_provenance'],
                run_metadata=manifest.get('run_metadata',{}),
                video_count=len(metrics),frame_count=frames,query_count=points,
                aggregation='equal_video_mean_over_defined_values',
                raw_trackon_comparison=comparison,
                metrics=means,defined_video_counts=counts,per_video=per_video)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-manifest',required=True)
    parser.add_argument('--prediction-manifest',required=True)
    parser.add_argument('--split',required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    result=evaluate(args.dataset_manifest,args.prediction_manifest,args.split)
    output=Path(args.output)
    output.parent.mkdir(parents=True,exist_ok=True)
    # Do not overwrite an earlier experiment's evidence.
    with output.open('x') as stream:
        json.dump(result,stream,indent=2,allow_nan=False)
        stream.write('\n')
    print(json.dumps(result['metrics'],allow_nan=False))


if __name__=='__main__':
    main()
