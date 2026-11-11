"""Display both mean error and frequency of harmful C3 proposals on paired rows."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--audit', required=True, type=Path)
p.add_argument('--output', required=True, type=Path)
args = p.parse_args()
audit_bytes = args.audit.read_bytes()
report = json.loads(audit_bytes)
assert report['state'] == 'complete_published_subset'
rows = report['rows']
stats = [row['candidates'][1] for row in rows]
assert all(s['name'] == 'C3' and s['common_valid_visible_rows'] > 0 for s in stats)
for s in stats:
    np.testing.assert_allclose(s['old']['raw_mean_error_256px'], s['new']['raw_mean_error_256px'], atol=1e-5, rtol=0)
x = np.arange(len(rows))
fig, axes = plt.subplots(1, 2, figsize=(14, 5.3), layout='constrained')
for offset, label, values, color in [
    (-.25, 'Raw 2D, same rows', [s['old']['raw_mean_error_256px'] for s in stats], '#777777'),
    (0, 'C3: bootstrap cache', [s['old']['mean_error_256px'] for s in stats], '#2678b2'),
    (.25, 'C3: learned-policy cache', [s['new']['mean_error_256px'] for s in stats], '#e58223'),
]:
    axes[0].bar(x + offset, values, .25, label=label, color=color)
axes[0].set(title='C3 candidate error versus raw on paired rows', ylabel='Mean error (256-scaled pixels; lower is better)')
for key, offset, label, color in [('old', -.18, 'Bootstrap cache', '#2678b2'), ('new', .18, 'Learned-policy cache', '#e58223')]:
    bars = axes[1].bar(x + offset, [100 * s[key]['worse_than_raw_by_1px_fraction'] for s in stats], .36, label=label, color=color)
    axes[1].bar_label(bars, fmt='%.1f', fontsize=8)
axes[1].set(title='C3 proposals worse than raw by >1px', ylabel='Percentage of paired visible rows (%)', ylim=(0, 100))
for ax in axes:
    ax.set_xticks(x, [r['scene'].split('-')[0] for r in rows], rotation=25, ha='right')
    ax.legend(fontsize=8)
    ax.grid(axis='y', alpha=.15)
fig.suptitle(f'{len(rows)} published TRAIN videos | identical rows valid in both executions\nCandidate diagnostic only; not fusion performance or isolated feedback causality')
args.output.mkdir(parents=True, exist_ok=False)
fig.savefig(args.output / 'paired_c3_risk.png', dpi=160)
plt.close(fig)
(args.output / 'binding.json').write_text(json.dumps(dict(
    audit=str(args.audit.resolve()), audit_sha256=hashlib.sha256(audit_bytes).hexdigest(),
    script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    scope=report['scope']), indent=2) + '\n')
