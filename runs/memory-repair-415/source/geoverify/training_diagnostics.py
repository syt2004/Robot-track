"""Plot recorded training losses; these are not held-out accuracy metrics."""
import argparse
import json
from pathlib import Path

import numpy as np


def render(history_path, output, *, label, track_weight=10.):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from .datasets import file_sha256

    history_path = Path(history_path)
    records = json.loads(history_path.read_text())
    if not records or [r['epoch'] for r in records] != list(range(1, len(records)+1)):
        raise ValueError('Expected complete consecutive epoch history')
    names = ('rank', 'track', 'visibility', 'reliability', 'excess')
    losses = np.array([r['loss'] for r in records], dtype=float)
    components = np.array([[r['components'][k] for k in names] for r in records], dtype=float)
    if not np.isfinite(losses).all() or not np.isfinite(components).all():
        raise ValueError('Non-finite training loss')
    if not np.isfinite(track_weight) or track_weight <= 0:
        raise ValueError('Expected finite positive track weight')
    if not np.allclose(losses, components @ np.array([1., track_weight, 1., 1., track_weight]), rtol=1e-5, atol=1e-7):
        raise ValueError('Total loss disagrees with weighted components')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    epochs = np.arange(1, len(records)+1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    axes[0].plot(epochs, losses, marker='o', markersize=3)
    axes[0].set_title('Total training objective')
    for i, name in enumerate(names):
        factor = track_weight if name in ('track','excess') else 1
        axes[1].plot(epochs, components[:, i]*factor, label=f'{track_weight:g} × {name}' if name in ('track','excess') else name)
    axes[1].set_title('Weighted objective components')
    axes[1].legend()
    for ax in axes:
        ax.set(xlabel='Completed epoch', ylabel='Mean loss per optimizer step')
        ax.grid(alpha=.25)
    fig.suptitle(label + '\nTraining data only — generalization requires separate held-out evaluation', fontsize=11)
    fig.savefig(output/'training_curves.png', dpi=160)
    plt.close(fig)
    report = dict(history_path=str(history_path.resolve()), history_sha256=file_sha256(history_path),
                  label=label, epochs=len(records), first_loss=float(losses[0]), last_loss=float(losses[-1]),
                  aggregation='Unweighted mean over optimizer steps; not point-weighted',
                  held_out_evaluation=False, track_weight=track_weight)
    (output/'training_diagnostics.json').write_text(json.dumps(report, indent=2)+'\n')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--history', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--label', required=True, help='Dataset/run identity; mark synthetic fixtures explicitly')
    parser.add_argument('--track-weight', type=float, default=10.)
    args = parser.parse_args()
    render(args.history, args.output, label=args.label, track_weight=args.track_weight)
