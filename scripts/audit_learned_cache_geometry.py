"""Compare paired TRAIN geometry candidates; never fit or alter the live run."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    exp = args.experiment.resolve()
    contract = json.loads((exp / 'experiment.json').read_text())
    spec = importlib.util.spec_from_file_location('locked_alignment', exp / 'merge_aligned.py')
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    old_path = Path(contract['training_manifest_control'])
    new_path = Path(contract['learned_cache']) / 'progress.json'
    old_bytes, new_bytes = old_path.read_bytes(), new_path.read_bytes()
    old, new = json.loads(old_bytes), json.loads(new_bytes)
    teacher = contract['teacher_checkpoint_sha256']
    assert new['teacher_checkpoint_sha256'] == teacher
    groups = {}
    for entry in new['shards']:
        groups.setdefault(entry['source_sha256'], []).append(entry)
    rows = []
    for source, entries in groups.items():
        dense, seen = helper.load_scene(new_path, entries, teacher)
        consumed = np.zeros_like(seen)
        chunks = {'old': [], 'new': [], 'old_valid': [], 'new_valid': [], 'gt': []}
        labeled = 0
        for entry in old['shards']:
            if entry['source_sha256'] != source:
                continue
            path = (old_path.parent / entry['path']).resolve()
            assert helper.file_sha256(path) == entry['sha256']
            data = helper.validate_arrays(dict(np.load(path, allow_pickle=False)), len(helper.FEATURE_NAMES))
            t, q = np.array(entry['frame_ids'])[:, None], np.array(entry['query_ids'])[None, :]
            assert seen[t, q].all() and not consumed[t, q].any()
            consumed[t, q] += 1
            aligned = {key: value[t, q] for key, value in dense.items()}
            for key in ('target_visible', 'target_valid'):
                np.testing.assert_array_equal(data[key], aligned[key])
            np.testing.assert_allclose(data['target_xy'], aligned['target_xy'], atol=0, rtol=0, equal_nan=True)
            np.testing.assert_allclose(data['candidates'][..., 0, :], aligned['candidates'][..., 0, :], atol=1e-5, rtol=0)
            use = data['target_valid'].astype(bool) & data['target_visible'].astype(bool)
            labeled += int(data['target_valid'].sum())
            for prefix, values in [('old', data), ('new', aligned)]:
                chunks[prefix].append(values['candidates'][use])
                chunks[prefix + '_valid'].append(values['valid'][use].astype(bool))
            chunks['gt'].append(data['target_xy'][use])
        np.testing.assert_array_equal(consumed, seen)
        arrays = {key: np.concatenate(value) for key, value in chunks.items()}
        scale = np.array([256 / entries[0]['width'], 256 / entries[0]['height']])
        errors = {key: np.linalg.norm((arrays[key] - arrays['gt'][:, None, :]) * scale, axis=-1) for key in ('old', 'new')}
        row = dict(scene=entries[0]['scene'], source_sha256=source, labeled_point_frames=labeled,
                   visible_labeled_point_frames=len(arrays['gt']), candidates=[])
        for index, name in [(1, 'C2'), (2, 'C3')]:
            common = arrays['old_valid'][:, index] & arrays['new_valid'][:, index]
            item = dict(name=name, common_valid_visible_rows=int(common.sum()),
                        old_valid_visible_rows=int(arrays['old_valid'][:, index].sum()),
                        new_valid_visible_rows=int(arrays['new_valid'][:, index].sum()))
            for key in ('old', 'new'):
                e, base = errors[key][common, index], errors[key][common, 0]
                assert np.isfinite(e).all() and np.isfinite(base).all()
                item[key] = dict(mean_error_256px=float(e.mean()), raw_mean_error_256px=float(base.mean()),
                                 better_than_raw_by_1px_fraction=float(np.mean(e < base - 1)),
                                 worse_than_raw_by_1px_fraction=float(np.mean(e > base + 1))) if len(e) else None
            row['candidates'].append(item)
        rows.append(row)
    report = dict(state='complete_published_subset', videos=len(rows), rows=rows,
                  control_manifest_sha256=hashlib.sha256(old_bytes).hexdigest(),
                  learned_progress_snapshot_sha256=hashlib.sha256(new_bytes).hexdigest(),
                  teacher_checkpoint_sha256=teacher,
                  scope='TRAIN only. C2/C3 compared on identical visible labeled rows valid in BOTH executions. Errors in 256-scaled image pixels. Async timing and learned feedback both differ; not a causal isolation, model gain, or DEV result.')
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / 'progress_snapshot.json').write_bytes(new_bytes)
    (args.output / 'audit.json').write_text(json.dumps(report, indent=2) + '\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    for index, ax in enumerate(axes):
        x = np.arange(len(rows))
        for key, offset, label in [('old', -.18, 'Bootstrap cache'), ('new', .18, 'Learned-policy cache')]:
            values = [r['candidates'][index][key]['mean_error_256px'] if r['candidates'][index][key] else np.nan for r in rows]
            ax.bar(x + offset, values, .36, label=label)
        ax.set_xticks(x, [r['scene'].split('-')[0] for r in rows], rotation=25, ha='right')
        ax.set_title(f'C{index + 2}: common valid, visible TRAIN rows')
        ax.set_ylabel('Mean candidate error (256-scaled pixels; lower is better)')
        ax.legend()
    fig.suptitle('Paired candidate diagnostic — not an evaluation of the trained model')
    fig.savefig(args.output / 'paired_geometry.png', dpi=160)
    plt.close(fig)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
