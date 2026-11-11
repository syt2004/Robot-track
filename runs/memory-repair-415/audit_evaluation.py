"""Audit saved complete-video results, without modifying predictions or models."""
from pathlib import Path
import argparse
import hashlib
import json
import sys

O = Path(__file__).resolve().parent
sys.path.insert(0, str(O / 'source'))
import numpy as np
import torch
from geoverify.calibration import GeometryCalibrationHead
from geoverify.datasets import iter_evaluation
from geoverify.metrics import tapvid_first
from geoverify.visibility_stream import RawVisibilityStream, replay_raw_visibility


def audit(case, split):
    cfg = json.loads((case / 'inference.json').read_text())
    manifest = json.loads((case / 'output/predictions.json').read_text())
    saved_metrics = json.loads((case / 'output/metrics.json').read_text())
    entry = manifest['videos'][0]
    path = case / 'output' / entry['path']
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert digest == entry['sha256']
    a = dict(np.load(path, allow_pickle=False))
    s = next(iter_evaluation(case / 'manifest.json', split))
    head = GeometryCalibrationHead('scalar_only').eval()
    head.load_state_dict(torch.load(cfg['calibrator_checkpoint'], map_location='cpu', weights_only=True)['model'])
    mask = a['processed'].astype(bool)
    keys = ['calibration_features', 'visual_logits', 'candidates', 'candidate_visibility', 'candidate_valid']
    inputs = {k: a[k][mask] for k in keys}
    maximum = 0.
    with torch.no_grad():
        for start in range(0, int(mask.sum()), 2048):
            b = [torch.as_tensor(inputs[k][start:start+2048]).bool() if k == 'candidate_valid'
                 else torch.as_tensor(inputs[k][start:start+2048]).float() for k in keys]
            out = head(*b, torch.zeros((*b[4].shape, 256)))
            delta = float(np.max(np.abs(a['xy'][mask][start:start+2048] - out['xy'].numpy())))
            maximum = max(maximum, delta)
            assert delta < .002, delta
    stream = RawVisibilityStream(cfg['visibility_checkpoint'], cfg['visibility_sha256'], cfg['baseline_provenance'])
    w, h = map(int, a['image_size'])
    limit = int(a['max_queries_per_cohort'])
    shadow = cfg.get('memory_repair', {}).get('history_reference') == 'independent_shadow_history_v1'
    if shadow: assert a['shadow_inputs_recorded'][mask].all()
    raw_xy = a['shadow_xy'] if shadow else a['candidates'][:, :, 0]
    raw_visibility = a['shadow_visibility'] if shadow else a['candidate_visibility'][:, :, 0]
    visibility = replay_raw_visibility(stream, raw_xy, raw_visibility,
                                      a['timestamps'], a['query_txy'], w, h, None if limit == -1 else limit)
    np.testing.assert_allclose(visibility, a['visibility'], atol=1e-5, rtol=1e-6)
    metric = tapvid_first(a['query_txy'], s.target_xy, s.target_visible, a['xy'], a['visible'], w, h)
    assert abs(metric['average_jaccard'] - saved_metrics['metrics']['average_jaccard']) < 1e-12
    events_path = case / 'memory_events.jsonl'
    events = [json.loads(line) for line in events_path.read_text().splitlines()] if events_path.exists() else []
    cohorts = []
    for start in np.unique(a['query_txy'][:, 0]).astype(int):
        ids = np.flatnonzero(a['query_txy'][:, 0] == start)
        size = len(ids) if limit == -1 else limit
        cohorts.extend((start, ids[j:j+size]) for j in range(0, len(ids), size))
    seen = set()
    for e in events:
        assert 1 <= e['cohort'] <= len(cohorts)
        start, ids = cohorts[e['cohort'] - 1]
        assert e['completed_timestamp'] <= e['applied_timestamp']
        assert e['effective_frame'] == e['received_frame'] + 1
        assert 0 <= e['geometry_anchor_frame'] <= e['geometry_target_frame'] <= e['received_frame']
        assert e['received_frame'] + start < len(s.rgb)
        assert all(0 <= q < len(ids) for q in e['query_ids'])
        assert np.isfinite(e['max_change']) and e['max_change'] >= 0
        if shadow: assert np.isfinite(e['max_shadow_relative_change']) and e['max_shadow_relative_change'] <= .25001
        assert e['verification_protocol']=='paired_historical_verifier_margin_v1'
        assert 0 <= e['slots'] <= e['verified_slots'] <= e['geometry_valid_slots']
        if e['verified_slots']:assert e['verification_window_available'] and e['minimum_verified_margin']>1e-6
        if not e['verification_window_available']:assert e['slots']==0 and e['max_change']==0
        key = (e['cohort'], e['job_id'])
        assert key not in seen
        seen.add(key)
    bounds_path = case / 'state_bounds.jsonl'
    if shadow:
        bounds = [json.loads(line) for line in bounds_path.read_text().splitlines()]
        assert bounds
        expected = {(c+1,t) for c,(start,ids) in enumerate(cohorts) for t in range(len(s.rgb)-start)}
        observed = {(b['cohort'],b['frame']) for b in bounds}
        assert len(bounds) == len(observed) and observed == expected, 'Incomplete or duplicate frame-bound audit'
        assert all(np.isfinite(b['max_shadow_relative_change']) and b['max_shadow_relative_change'] <= .25001 for b in bounds)
    result = dict(state='passed', video=s.name, tracks_sha256=digest, AJ=metric['average_jaccard'],
                  max_fusion_xy_difference=maximum, events=len(events),
                  effective_events=sum(e['slots'] > 0 and e['max_change'] > 0 for e in events),
                  scope='CPU fusion/visibility/metric recomputation and causal memory-event checks; not memory-OFF counterfactual')
    (case / 'audit.json').write_text(json.dumps(result, indent=2))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True)
    parser.add_argument('--split', default='dev')
    args = parser.parse_args()
    torch.set_num_threads(1)
    for status_path in sorted(Path(args.root).glob('group*/video_*/output/status.json')):
        if json.loads(status_path.read_text())['state'] == 'complete':
            print(json.dumps(audit(status_path.parents[1], args.split)), flush=True)
