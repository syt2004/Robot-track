"""Plot verified published no-geometry reruns against the complete013 control."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--experiment',type=Path,required=True)
p.add_argument('--control',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
args=p.parse_args();exp=args.experiment.resolve();control=args.control.resolve()
sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
baseline=json.loads((control/'comparison_013.json').read_text())
progress=json.loads((exp/'development/progress.json').read_text())
assert progress['dataset_manifest_sha256']==baseline['manifest_sha256']
assert sha(exp/'training/calibrator.pt')==sha(control/'training/calibrator.pt')
assert json.loads((exp/'source_manifest.json').read_text())==json.loads((control/'source_manifest.json').read_text())
rows=[]
for i,entry in enumerate(progress['videos']):
    path=exp/'development'/entry['path']
    assert sha(path)==entry['sha256']
    diagnostic=path.parent/'effect_diagnostic/effects.json'
    live=exp/'live_replay_audits'/f'video_{i:06d}.json'
    canonical=exp/'calibration_replay_audits'/f'video_{i:06d}.json'
    audit=canonical if canonical.exists() else live
    if not diagnostic.exists() or not audit.exists():continue
    a=json.loads(audit.read_text());effects=json.loads(diagnostic.read_text())
    assert a['state']=='passed' and a['tracks_sha256']==effects['prediction_sha256']==entry['sha256']
    assert a['checkpoint_sha256']==sha(exp/'training/calibrator.pt')
    assert a['no_geometry_age_invariance_rows']==a['processed_point_frames']
    with np.load(path,allow_pickle=False) as data:
        active=data['processed'].astype(bool)
        assert not data['candidate_valid'][...,1:3].any()
        np.testing.assert_array_equal(data['fusion_weights'][...,1:3][active],0.)
    name=entry['name'];raw=baseline['per_video']['raw'][name]['average_jaccard']
    assert effects['metrics']['raw']['average_jaccard']==raw
    rows.append(dict(video=name,raw_AJ=raw,full013_AJ=baseline['per_video']['full'][name]['average_jaccard'],
        no_geometry_same_head_AJ=effects['metrics']['full']['average_jaccard'],prediction_sha256=entry['sha256']))
assert rows,'No verified video results yet'
expected=len(baseline['per_video']['raw'])
complete=len(rows)==expected and json.loads((exp/'status.json').read_text())['state']=='complete'
report=dict(state='complete' if complete else 'partial',verified_videos=len(rows),expected_videos=expected,rows=rows,
    checkpoint_sha256=sha(exp/'training/calibrator.pt'),scope='Independent geometry-disabled reruns with identical frozen head. No separately retrained ablation model; changed feedback and timing are part of module removal. Not a full-method freeze candidate.')
if complete:
    official=json.loads((exp/'comparison.json').read_text())
    report['equal_video_mean_AJ']={key:float(np.mean([r[key] for r in rows])) for key in ['raw_AJ','full013_AJ','no_geometry_same_head_AJ']}
    np.testing.assert_allclose(report['equal_video_mean_AJ']['no_geometry_same_head_AJ'],official['equal_video_mean_AJ']['ablation'],atol=0,rtol=0)
x=np.arange(len(rows));fig,ax=plt.subplots(figsize=(10,5),layout='constrained')
for offset,key,label,color in [(-.18,'full013_AJ','Full013','#b76b61'),(.18,'no_geometry_same_head_AJ','Same head, geometry disabled','#397d9c')]:
    bars=ax.bar(x+offset,[100*(r[key]-r['raw_AJ']) for r in rows],.36,label=label,color=color)
    ax.bar_label(bars,fmt='%+.3f',padding=4)
ax.axhline(0,color='black',linewidth=.8)
names={'gear_v2_6-basket_front_right_rgb_img-1':'Gear','10790317943435558913':'Shape sorting','shoe_black-sawyer_camera2_rgb_img':'Shoe'}
ax.set_xticks(x,[names.get(r['video'],r['video']) for r in rows]);ax.set(ylabel='AJ difference from raw (percentage points)',title=f'Independent robot DEV reruns | {len(rows)}/{expected} verified videos\n'+('All fixed videos complete; identical trained head' if complete else 'Partial results: no overall mean or full-set conclusion'))
ax.legend();ax.grid(axis='y',alpha=.2);ax.margins(y=.22)
args.output.mkdir(parents=True,exist_ok=False)
fig.savefig(args.output/'independent_ablation.png',dpi=160);plt.close(fig)
(args.output/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
