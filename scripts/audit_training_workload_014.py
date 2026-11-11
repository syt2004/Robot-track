"""Read real 014 shards and time fresh-model updates without altering its checkpoint."""
from pathlib import Path
import hashlib
import json
import math
import sys
import time

import numpy as np
import torch

root = Path(__file__).resolve().parents[1]
exp = root / 'runs/learned-rollout-head-014'
sys.path.insert(0, str(exp / 'source'))
from geoverify.calibration import GeometryCalibrationHead, calibration_loss
from geoverify.point_sampling import labeled_rows, epoch_indices
from geoverify.cache import validate_arrays
from geoverify.feature_schema import FEATURE_NAMES

torch.set_num_threads(2)
manifest = json.loads((exp / 'training_manifest.json').read_text())
tables, scenes = [], {}
for entry in manifest['shards']:
    path = (exp / entry['path']).resolve()
    assert hashlib.sha256(path.read_bytes()).hexdigest() == entry['sha256']
    with np.load(path, allow_pickle=False) as cache:
        data = validate_arrays(cache, len(FEATURE_NAMES))
        rows = labeled_rows(data)
    n = len(rows['target_valid'])
    scenes[entry['scene']] = scenes.get(entry['scene'], 0) + n
    rows['image_diagonal'] = np.full(n, math.hypot(entry['width'], entry['height']), np.float32)
    tables.append(rows)
table = {k: np.concatenate([t[k] for t in tables]) for k in tables[0]}
checkpoint = torch.load(exp / 'training/calibrator.pt', map_location='cpu', weights_only=True)
torch.manual_seed(0)
model = GeometryCalibrationHead()
changes = {k: float((checkpoint['model'][k] - v.detach()).abs().max()) for k, v in model.named_parameters()}
model.feature_mean.copy_(checkpoint['model']['feature_mean'])
model.feature_std.copy_(checkpoint['model']['feature_std'])
optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
rng = np.random.default_rng(0)
batches = list(epoch_indices(len(table['target_valid']), 128, rng))
assert len(np.unique(np.concatenate(batches))) == len(table['target_valid'])
times = []
for i, ids in enumerate(batches[:320]):
    start = time.perf_counter()
    data = {k: torch.as_tensor(v[ids]) for k, v in table.items()}
    result = model(data['features'].float(), data['visual_logits'].float(), data['candidates'].float(), data['candidate_visibility'].float(), data['valid'])
    loss, _ = calibration_loss(result, data['candidates'].float(), data['valid'].bool(), data['target_xy'].float(), data['target_visible'], data['target_valid'], data['image_diagonal'])
    assert torch.isfinite(loss)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
    optimizer.step()
    if i >= 20:
        times.append(time.perf_counter() - start)
report = dict(
    state='complete', trainable_parameters=sum(p.numel() for p in model.parameters()),
    scene_labeled_point_frames=scenes, total_labeled_point_frames=len(table['target_valid']),
    shards=len(manifest['shards']), batch_feature_shape=list(data['features'].shape),
    table_feature_shape=list(table['features'].shape), steps_per_epoch=len(batches),
    checkpoint_epoch=checkpoint['epoch'], resumed_from=checkpoint['resumed_from_sha256'],
    actual_optimizer_step_values=sorted({int(s['step'].item()) for s in checkpoint['optimizer']['state'].values()}),
    parameter_change_from_initialization=changes, timing_threads=torch.get_num_threads(),
    warmup_steps=20, measured_steps=len(times), measured_total_seconds=sum(times),
    mean_step_seconds=float(np.mean(times)), median_step_seconds=float(np.median(times)),
    projected_22200_update_seconds=float(np.mean(times) * 22200),
    checkpoint_sha256=hashlib.sha256((exp / 'training/calibrator.pt').read_bytes()).hexdigest(),
    scope='Independent 300-step CPU timing on actual cached point-frame batches after 20 warmup updates. Fresh temporary model, no checkpoint saved or promoted. Projection excludes data preparation, normalization, checkpoint saves and plotting; not a full 20-epoch rerun. Point-frames are not independent videos.')
out = root / 'runs/training-workload-audit-014'
out.mkdir(exist_ok=False)
(out / 'audit.json').write_text(json.dumps(report, indent=2) + '\n')
(out / 'script.py').write_bytes(Path(__file__).read_bytes())
print(json.dumps(report, indent=2))
