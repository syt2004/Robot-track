"""Show actual rigid-refined candidates; not a counterfactual or accuracy test."""
import argparse
import hashlib
import json
from pathlib import Path

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--experiment', type=Path, required=True)
args = p.parse_args()
exp = args.experiment.resolve()
meta = json.loads((exp/'input.json').read_text())
runtime = json.loads((exp/'rigid/runtime.json').read_text())
track = exp/'rigid/tracks.npz'
assert runtime['state'] == 'complete'
assert sha(track) == runtime['artifacts']['tracks.npz']
assert sha(meta['video']) == meta['video_sha256']
a = dict(np.load(track, allow_pickle=False))
mask = a['rigid_correction_mask'].astype(bool)
assert mask.any() and np.all(a['candidate_valid'][..., 2][mask])
indices = np.argwhere(mask)
distance = np.linalg.norm(a['xy']-a['candidates'][..., 0, :], axis=-1)
chosen = [('First affected point-frame', indices[0]),
          ('Middle affected point-frame', indices[len(indices)//2]),
          ('Largest fused/raw separation', indices[np.argmax(distance[mask])])]
cap = cv2.VideoCapture(meta['video'])
frames = {}
for t in range(int(mask.shape[0])):
    ok, frame = cap.read()
    assert ok, f'Decode failed at {t}'
    if t in {int(row[1][0]) for row in chosen}:
        frames[t] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
cap.release()
fig, axes = plt.subplots(3, 2, figsize=(12, 12), constrained_layout=True)
rows = []
for i, (selection, (t, q)) in enumerate(chosen):
    xy = np.array([a['candidates'][t,q,0], a['candidates'][t,q,2], a['xy'][t,q]])
    assert np.isfinite(xy).all()
    for ax in axes[i]:
        ax.imshow(frames[int(t)])
        for point, color, marker, label in zip(xy, ['cyan','magenta','orange'], ['o','x','+'],
                ['Raw C1', 'Rigid-refined delayed C3', 'Fused output']):
            ax.scatter(*point, c=color, marker=marker, s=85, label=label)
        ax.plot(xy[:,0], xy[:,1], color='white', linewidth=.7, alpha=.6)
    axes[i,0].set_title(f'{selection}: frame {t}, query {q}, object {a["object_ids"][q]}')
    axes[i,0].axis('off')
    center = (xy.min(0)+xy.max(0))/2
    radius = max(25., float(np.ptp(xy,axis=0).max())/2+12)
    axes[i,1].set(xlim=(center[0]-radius,center[0]+radius),
                  ylim=(center[1]+radius,center[1]-radius),
                  title=f'C3 weight {a["fusion_weights"][t,q,2]:.4f}; fused/raw {distance[t,q]:.3f}px')
    axes[i,1].legend(fontsize=8)
    rows.append(dict(selection=selection, frame=int(t), query=int(q),
        object_id=int(a['object_ids'][q]), raw_xy=xy[0].tolist(),
        refined_c3_xy=xy[1].tolist(), fused_xy=xy[2].tolist(),
        c3_weight=float(a['fusion_weights'][t,q,2]),
        fused_raw_distance_pixels=float(distance[t,q]),
        geometry_target_frame=int(a['geometry_target_frame'][t,q])))
fig.suptitle(f'{exp.name} optional rigid branch on original robot RGB | {int(mask.sum())} affected point-frames\n'
    'C3 is refined BEFORE fusion. No GT or no-rigid rerun: separation is NOT improvement.')
out = exp/'candidate_examples'
out.mkdir(exist_ok=False)
fig.savefig(out/'rigid_candidates.png', dpi=150)
plt.close(fig)
(out/'selection.json').write_text(json.dumps(dict(tracks_sha256=sha(track),
    video_sha256=sha(meta['video']), affected_point_frames=int(mask.sum()),
    affected_queries=np.unique(indices[:,1]).tolist(), records=rows,
    scope='Actual recorded candidates only. Pre-rigid C3 was not saved; no before/after rigid causal claim.'), indent=2)+'\n')
print(json.dumps(rows, indent=2))
