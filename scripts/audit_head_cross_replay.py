"""Factor frozen head changes from recorded-input changes, without online reruns."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import torch


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def forward_definition(path):
    tree = ast.parse(path.read_text())
    return [ast.dump(node, include_attributes=False) for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.ClassDef))
            and node.name in ['canonical_features', 'GeometryCalibrationHead']]


p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--reference', required=True, type=Path)
p.add_argument('--candidate', required=True, type=Path)
p.add_argument('--video', required=True, type=int)
p.add_argument('--output', required=True, type=Path)
args = p.parse_args()
root = Path(__file__).resolve().parents[1]
experiments = [args.reference.resolve(), args.candidate.resolve()]
assert not args.output.exists()
assert forward_definition(experiments[0]/'source/geoverify/calibration.py') == forward_definition(experiments[1]/'source/geoverify/calibration.py'), 'Cross replay requires identical forward computation'
sys.path.insert(0, str(experiments[0]/'source'))
from geoverify.calibration import GeometryCalibrationHead
from geoverify.datasets import iter_evaluation
from geoverify.metrics import tapvid_first

torch.set_num_threads(2)
manifest = root/'runs/robot-development-001/manifest.json'
sample = next(s for i, s in enumerate(iter_evaluation(manifest, 'dev')) if i == args.video)
models = []
checkpoints = []
for exp in experiments:
    for name, digest in json.loads((exp/'source_manifest.json').read_text()).items():
        assert sha(exp/'source'/name) == digest
    config = json.loads((exp/'inference.json').read_text())
    checkpoint = Path(config['calibrator_checkpoint'])
    assert sha(checkpoint) == config['calibrator_sha256']
    state = torch.load(checkpoint, map_location='cpu', weights_only=True)
    assert state['manifest_sha256'] == sha(exp/'training_manifest.json')
    assert state['head_architecture'] == config['head_architecture']
    model = GeometryCalibrationHead().eval()
    model.load_state_dict(state['model'], strict=True)
    models.append(model)
    checkpoints.append(sha(checkpoint))

rows = []
for input_index, exp in enumerate(experiments):
    progress = json.loads((exp/'development/progress.json').read_text())
    assert progress['dataset_manifest_sha256'] == sha(manifest)
    entry = progress['videos'][args.video]
    assert entry['name'] == sample.name
    track = exp/'development'/entry['path']
    assert sha(track) == entry['sha256']
    a = dict(np.load(track, allow_pickle=False))
    np.testing.assert_array_equal(a['query_txy'], sample.query_txy)
    mask = a['processed'].astype(bool)
    np.testing.assert_array_equal(mask, a['calibration_inputs_recorded'])
    keys = ['calibration_features','visual_logits','candidates','candidate_visibility','candidate_valid']
    flat = [a[k][mask] for k in keys]
    for head_index, model in enumerate(models):
        batches = {k: [] for k in ['xy','visibility','confidence','weights']}
        with torch.inference_mode():
            for start in range(0, int(mask.sum()), 2048):
                inputs = [torch.as_tensor(v[start:start+2048]).bool() if j == 4
                          else torch.as_tensor(v[start:start+2048]).float() for j, v in enumerate(flat)]
                result = model(*inputs)
                for key in batches:
                    batches[key].append(result[key].numpy())
        outputs = {k: np.concatenate(v) for k, v in batches.items()}
        maxdiff = {}
        if input_index == head_index:
            for key, saved in [('xy','xy'),('visibility','visibility'),('confidence','confidence'),('weights','fusion_weights')]:
                actual, expected = outputs[key], a[saved][mask]
                np.testing.assert_allclose(actual, expected, atol=.002 if key == 'xy' else 1e-5, rtol=1e-6)
                maxdiff[key] = float(np.max(np.abs(actual-expected)))
        xy = a['xy'].copy()
        xy[mask] = outputs['xy']
        visibility = a['visibility'].copy()
        visibility[mask] = outputs['visibility']
        visible = (visibility >= float(a['visibility_threshold'])) & mask
        metrics = tapvid_first(sample.query_txy, sample.target_xy, sample.target_visible,
                              xy, visible, sample.rgb.shape[2], sample.rgb.shape[1])
        rows.append(dict(input_experiment=exp.name, head_experiment=experiments[head_index].name,
            prediction_sha256=sha(track), checkpoint_sha256=checkpoints[head_index],
            same_head_saved_replay_maxdiff=maxdiff or None, metrics=metrics,
            mean_geometry_weight=float(outputs['weights'][:,1:3].sum(-1).mean())))
report = dict(state='complete', video=sample.name, dataset_manifest_sha256=sha(manifest), rows=rows,
    scope='2x2 saved-input CPU replay. Diagonals verified against actual output. Off-diagonals reuse original candidates, visual logits, features and feedback histories; NOT independently run online ablations or deployable accuracy. Input differences also include asynchronous geometry timing; not attributable solely to feedback.')
args.output.mkdir(parents=True)
(args.output/'audit.json').write_text(json.dumps(report, indent=2)+'\n')
print(json.dumps(report, indent=2))
