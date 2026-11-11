"""Robot RGB evidence for delayed geometry candidate errors, before fusion."""
from pathlib import Path
import sys, json, argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))
from geoverify.datasets import iter_evaluation, file_sha256

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--experiment', required=True, type=Path)
p.add_argument('--output', required=True, type=Path)
args = p.parse_args()
manifest = root/'runs/robot-development-001/manifest.json'
predictions = json.loads((args.experiment/'development/predictions.json').read_text())
assert predictions['dataset_manifest_sha256'] == file_sha256(manifest)
fig, axes = plt.subplots(3, 3, figsize=(15, 12), constrained_layout=True)
records = []
colors = ['lime', 'cyan', 'magenta', 'orange']
labels = ['GT', 'Raw C1', 'Projected C3', 'Fused']
for i, (sample, entry) in enumerate(zip(iter_evaluation(manifest, 'dev'), predictions['videos'], strict=True)):
    assert sample.name == entry['name']
    path = args.experiment/'development'/entry['path']
    assert file_sha256(path) == entry['sha256']
    with np.load(path, allow_pickle=False) as archive:
        a = dict(archive)
    use = (np.arange(len(sample.rgb))[:, None] > sample.query_txy[None, :, 0]) & sample.target_visible & a['candidate_valid'][..., 2]
    indices = np.argwhere(use)
    error = np.linalg.norm(a['candidates'] - sample.target_xy[..., None, :], axis=-1)
    delta = error[..., 2] - error[..., 0]
    chosen = [indices[len(indices)//2], indices[np.argmax(delta[use])]]
    t, q = chosen[1]
    axes[i, 0].imshow(sample.rgb[t])
    axes[i, 0].scatter(*sample.target_xy[t, q], s=90, facecolors='none', edgecolors='lime')
    axes[i, 0].set(title=f'{sample.name[:29]}\nWorst C3 error increase: frame {t}, query {q}')
    axes[i, 0].axis('off')
    for column, ((t, q), selection) in enumerate(zip(chosen, ['Median valid point-frame', 'Maximum C3 error increase']), 1):
        xy = np.array([sample.target_xy[t, q], a['candidates'][t, q, 0], a['candidates'][t, q, 2], a['xy'][t, q]])
        assert np.isfinite(xy).all()
        ax = axes[i, column]
        ax.imshow(sample.rgb[t])
        for point, color, label, marker in zip(xy, colors, labels, ['*', 'o', 'x', '+']):
            ax.scatter(*point, c=color, label=label, marker=marker, s=90)
        for j in [1, 2, 3]:
            ax.plot(xy[[0, j], 0], xy[[0, j], 1], color=colors[j], alpha=.7, linewidth=1)
        lo, hi = xy.min(0)-20, xy.max(0)+20
        center = (lo+hi)/2
        radius = max(40, float((hi-lo).max())/2)
        ax.set_xlim(center[0]-radius, center[0]+radius)
        ax.set_ylim(center[1]+radius, center[1]-radius)
        ax.set_title(f'{selection}: t={t}, q={q}\nC1 {error[t,q,0]:.2f}px; C3 {error[t,q,2]:.2f}px')
        ax.legend(fontsize=7, loc='lower right')
        records.append(dict(video=sample.name, prediction_sha256=file_sha256(path), selection=selection,
            frame=int(t), query=int(q), positions=dict(zip(labels, xy.tolist())),
            raw_error_pixels=float(error[t,q,0]), projected_error_pixels=float(error[t,q,2]),
            fused_error_pixels=float(np.linalg.norm(a['xy'][t,q]-sample.target_xy[t,q])),
            projected_candidate_weight=float(a['fusion_weights'][t,q,2]),
            geometry_target_frame=int(a['geometry_target_frame'][t,q]),
            recorded_geometry_age=float(a['geometry_age'][t,q]),
            geometry_weight=float(a['fusion_weights'][t,q,1:3].sum())))
args.output.mkdir(parents=True, exist_ok=False)
fig.suptitle(args.experiment.name+' robot DEV: delayed projected candidates, not raw point-cloud accuracy\nMedian valid point-frame plus explicitly selected worst error increase; not representative averages')
fig.savefig(args.output/'candidate_failures.png', dpi=150)
(args.output/'selection.json').write_text(json.dumps(dict(manifest_sha256=file_sha256(manifest), records=records,
    experiment=str(args.experiment.resolve()),
    scope='Posthoc visible GT diagnostic; selected examples do not replace all-video aggregate statistics. RGB is original dataset; geometry endpoints shown are delayed corrected candidates, not unmodified Track4World output.'), indent=2)+'\n')
print(json.dumps(records, indent=2))
