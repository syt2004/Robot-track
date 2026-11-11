"""Posthoc visible-GT candidate quality; oracle is diagnostic, never deployable."""
from pathlib import Path
import argparse
import json
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))
from geoverify.datasets import iter_evaluation, file_sha256

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--experiment', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
args = p.parse_args()
manifest = root / 'runs/robot-development-001/manifest.json'
progress = json.loads((args.experiment / 'development/predictions.json').read_text())
assert progress['dataset_manifest_sha256'] == file_sha256(manifest)
rows = []
names = ['C1 raw', 'C2 geometry 2D', 'C3 projected 3D', 'C4 velocity']
for sample, entry in zip(iter_evaluation(manifest, 'dev'), progress['videos'], strict=True):
    assert sample.name == entry['name']
    path = args.experiment / 'development' / entry['path']
    assert file_sha256(path) == entry['sha256']
    with np.load(path, allow_pickle=False) as archive:
        a = dict(archive)
    mask = (np.arange(len(sample.rgb))[:, None] > sample.query_txy[None, :, 0]) & sample.target_visible
    assert a['processed'][mask].all()
    distance = np.linalg.norm(a['candidates'] - sample.target_xy[..., None, :], axis=-1)
    distance = np.where(a['candidate_valid'], distance, np.inf)
    raw = distance[..., 0]
    fused = np.linalg.norm(a['xy'] - sample.target_xy, axis=-1)
    oracle = distance.min(-1)
    selected = np.take_along_axis(distance, a['fusion_weights'].argmax(-1)[..., None], -1)[..., 0]
    candidates = []
    for j, name in enumerate(names):
        use = mask & a['candidate_valid'][..., j]
        delta = distance[..., j][use] - raw[use]
        candidates.append(dict(name=name, visible_valid_point_frames=int(use.sum()),
            mean_error_pixels=float(distance[..., j][use].mean()) if use.any() else None,
            raw_error_on_same_rows=float(raw[use].mean()) if use.any() else None,
            improves_over_raw_by_gt_1px=int((delta < -1).sum()),
            worsens_over_raw_by_gt_1px=int((delta > 1).sum())))
    row = dict(video=sample.name, prediction_sha256=file_sha256(path),
        visible_point_frames=int(mask.sum()), candidates=candidates,
        mean_error_pixels={k: float(v[mask].mean()) for k, v in
            [('raw', raw), ('fused', fused), ('max_weight_candidate', selected), ('gt_best_candidate', oracle)]},
        available_candidate_improves_raw_gt_1px=int((mask & (oracle < raw - 1)).sum()),
        fusion_hurts_raw_gt_1px=int((mask & (fused > raw + 1)).sum()),
        fusion_hurts_despite_better_candidate_gt_1px=int((mask & (fused > raw + 1) & (oracle < raw - 1)).sum()))
    rows.append(row)
args.output.mkdir(parents=True, exist_ok=False)
report = dict(state='complete', manifest_sha256=file_sha256(manifest), rows=rows,
    scope='All three fixed robot DEV videos; visible GT after query frame. GT oracle is posthoc diagnostic, not deployable or an independent ablation. Pixel errors are compared within video, never pooled across resolutions. No model or threshold fitted.')
(args.output/'audit.json').write_text(json.dumps(report, indent=2)+'\n')
fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
for ax, row in zip(axes, rows):
    values = row['mean_error_pixels']
    ax.bar(['Raw', 'Fused', 'Max weight', 'GT oracle'], list(values.values()))
    ax.set(title=row['video'][:25], ylabel='Visible GT mean error (pixels)')
    ax.tick_params(axis='x', rotation=20)
    ax.grid(axis='y', alpha=.2)
fig.suptitle('Saved candidates: geometry availability versus learned fusion\nGT oracle uses labels; diagnostic only, not achievable performance')
fig.savefig(args.output/'candidate_quality.png', dpi=160)
print(json.dumps(report, indent=2))
