"""Show both corrected and newly wrong visibility decisions on original RGB."""
from pathlib import Path
import argparse,json,sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root))
from geoverify.datasets import iter_evaluation,file_sha256
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--experiment',required=True,type=Path);p.add_argument('--video',required=True,type=int);args=p.parse_args()
exp=args.experiment.resolve();manifest=root/'runs/robot-development-001/manifest.json'
progress=json.loads((exp/'development/progress.json').read_text());assert progress['dataset_manifest_sha256']==file_sha256(manifest)
entry=progress['videos'][args.video];sample=next(s for i,s in enumerate(iter_evaluation(manifest,'dev')) if i==args.video);assert sample.name==entry['name']
path=exp/'development'/entry['path'];assert file_sha256(path)==entry['sha256']
a=dict(np.load(path,allow_pickle=False));raw=a['candidate_visibility'][...,0]>=float(a['visibility_threshold']);final=a['visible'].astype(bool);gt=sample.target_visible
mask=(np.arange(len(sample.rgb))[:,None]>sample.query_txy[None,:,0])&a['processed']
groups=[('Hidden GT: corrected to hidden',mask&~gt&raw&~final),
        ('Visible GT: corrected to visible',mask&gt&~raw&final),
        ('Visible GT: newly wrong hidden',mask&gt&raw&~final),
        ('Hidden GT: newly wrong visible',mask&~gt&~raw&final)]
fig,axes=plt.subplots(2,2,figsize=(12,10),layout='constrained');records=[]
for ax,(label,use) in zip(axes.flat,groups):
    ids=np.argwhere(use);record=dict(category=label,point_frames=len(ids))
    if not len(ids):ax.text(.5,.5,'No such cases',ha='center');ax.set_title(label);ax.axis('off');records.append(record);continue
    t,q=ids[len(ids)//2];ax.imshow(sample.rgb[t]);xy=np.array([a['candidates'][t,q,0],a['xy'][t,q]])
    assert np.isfinite(xy).all()
    ax.scatter(*xy[0],s=90,facecolors='none',edgecolors='cyan',label='Raw prediction')
    ax.scatter(*xy[1],s=90,c='orange',marker='+',label='Fused prediction')
    if gt[t,q]:ax.scatter(*sample.target_xy[t,q],s=100,c='lime',marker='*',label='Visible GT position')
    ax.set_title(f'{label} | {len(ids)} point-frames\nIndex-middle example: frame {t}, query {q}\nP(visible) raw {a["candidate_visibility"][t,q,0]:.3f} → fused {a["visibility"][t,q]:.3f}')
    ax.legend(fontsize=8);ax.axis('off')
    record.update(frame=int(t),query=int(q),gt_visible=bool(gt[t,q]),raw_xy=xy[0].tolist(),fused_xy=xy[1].tolist(),raw_visible_probability=float(a['candidate_visibility'][t,q,0]),fused_visible_probability=float(a['visibility'][t,q]))
    records.append(record)
fig.suptitle(exp.name+' | '+sample.name+'\nBoth improvements and regressions; predicted coordinates drawn even when hidden. Hidden GT coordinates not shown.')
out=path.parent/'visibility_cases';out.mkdir(exist_ok=False);fig.savefig(out/'visibility_changes.png',dpi=150);plt.close(fig)
(out/'selection.json').write_text(json.dumps(dict(prediction_sha256=file_sha256(path),dataset_manifest_sha256=file_sha256(manifest),records=records,scope='All evaluated point-frame counts; each example is index-middle within its class, not representative average or task-success evidence.'),indent=2)+'\n')
print(json.dumps(records,indent=2))
