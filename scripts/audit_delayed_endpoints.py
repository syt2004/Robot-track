"""Recover accepted endpoint deltas algebraically from saved delayed candidates.

This is not an independent Track4World rerun. Repeated uses of one endpoint
must reconstruct the same delta before target-frame error is interpreted.
"""
from pathlib import Path
import sys, json, argparse
import numpy as np
root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))
from geoverify.datasets import iter_evaluation, file_sha256
p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--experiment', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
p.add_argument('--candidate-index', type=int, choices=[1, 2], default=2)
args = p.parse_args()
config = json.loads((args.experiment/'inference.json').read_text())
tau = config['pipeline']['tau_seconds']
manifest = root/'runs/robot-development-001/manifest.json'
progress = json.loads((args.experiment/'development/predictions.json').read_text())
assert progress['dataset_manifest_sha256'] == file_sha256(manifest)
rows = []
for s, entry in zip(iter_evaluation(manifest, 'dev'), progress['videos'], strict=True):
    assert s.name == entry['name']
    path = args.experiment/'development'/entry['path']
    assert file_sha256(path) == entry['sha256']
    with np.load(path) as archive:
        a = dict(archive)
    records = {}
    consistency = 0.
    eligible = a['processed'] & a['candidate_valid'][..., args.candidate_index]
    if args.candidate_index == 2:
        eligible &= ~a['rigid_correction_mask']
    for t, q in np.argwhere(eligible):
        target = int(a['geometry_target_frame'][t,q])
        if not (t > s.query_txy[q,0] and s.target_visible[t,q] and s.target_visible[target,q]):
            continue
        delta = (a['candidates'][t,q,args.candidate_index]-a['candidates'][t,q,0])*np.exp(a['geometry_age'][t,q]/tau)
        endpoint = a['candidates'][target,q,0]+delta
        key = (int(q), int(a['geometry_anchor_frame'][t,q]), target)
        if key in records:
            consistency = max(consistency, float(np.max(np.abs(delta-records[key]['delta']))))
            continue
        records[key] = dict(delta=delta, target_raw=float(np.linalg.norm(a['candidates'][target,q,0]-s.target_xy[target,q])),
            target_projected=float(np.linalg.norm(endpoint-s.target_xy[target,q])),
            first_used_raw=float(np.linalg.norm(a['candidates'][t,q,0]-s.target_xy[t,q])),
            first_used_projected=float(np.linalg.norm(a['candidates'][t,q,args.candidate_index]-s.target_xy[t,q])))
    assert consistency < 1e-4, consistency
    values = list(records.values())
    summaries = {key: dict(mean=float(np.mean([v[key] for v in values])), median=float(np.median([v[key] for v in values])))
        for key in ['target_raw', 'target_projected', 'first_used_raw', 'first_used_projected']}
    rows.append(dict(video=s.name, prediction_sha256=file_sha256(path), unique_query_windows=len(values),
        maximum_repeated_delta_difference_pixels=consistency, errors_pixels=summaries,
        endpoint_worse_gt_1px=sum(v['target_projected']>v['target_raw']+1 for v in values),
        endpoint_better_gt_1px=sum(v['target_projected']<v['target_raw']-1 for v in values)))
args.output.parent.mkdir(parents=True, exist_ok=True)
with args.output.open('x') as f:
    json.dump(dict(state='complete', tau_seconds=tau, candidate_index=args.candidate_index, rows=rows,
        scope='Algebraic reconstruction of accepted geometry endpoints from saved candidates, not independent raw geometry verification. Fields named projected refer to selected candidate (C2 for index 1, C3 for index 2). Excludes rigid-corrected C3 rows, requires GT visible at target and use, deduplicates query/anchor/target windows. First-used means first eligible saved use. No fitted parameters.'), f, indent=2)
print(json.dumps(rows, indent=2))
