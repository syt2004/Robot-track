"""Posthoc position/visibility factorization of one completed robot video."""
from pathlib import Path
import sys,json,argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root))
from geoverify.datasets import iter_evaluation,file_sha256
from geoverify.metrics import tapvid_first
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--experiment',required=True);p.add_argument('--video',type=int,required=True);args=p.parse_args();exp=Path(args.experiment).resolve();dev=root/'runs/robot-development-001';manifest=dev/'manifest.json';progress=json.loads((exp/'development/progress.json').read_text());entry=progress['videos'][args.video]
assert progress['dataset_manifest_sha256']==file_sha256(manifest)
s=next(s for i,s in enumerate(iter_evaluation(manifest,'dev')) if i==args.video);assert s.name==entry['name']
pred=exp/'development'/entry['path'];assert file_sha256(pred)==entry['sha256'];a=dict(np.load(pred));b=dict(np.load(dev/'baseline'/f'video_{args.video:06d}/tracks.npz'));mask=np.arange(len(s.rgb))[:,None]>s.query_txy[None,:,0]
np.testing.assert_allclose(a['candidates'][...,0,:][mask],b['xy'][mask],atol=1e-5,rtol=0)
variants=[('raw',b['xy'],b['visible']),('raw_xy_fused_visibility',b['xy'],a['visible']),('fused_xy_raw_visibility',a['xy'],b['visible']),('full',a['xy'],a['visible'])];rows={}
for name,xy,vis in variants:
 m=tapvid_first(s.query_txy,s.target_xy,s.target_visible,xy,vis,s.rgb.shape[2],s.rgb.shape[1]);gt=s.target_visible
 m['false_hidden_visible_gt']=int((mask&gt&~vis).sum());m['false_visible_hidden_gt']=int((mask&~gt&vis).sum());m['true_visible_gt']=int((mask&gt&vis).sum());m['true_hidden_gt']=int((mask&~gt&~vis).sum());rows[name]=m
out=pred.parent/'effect_diagnostic';out.mkdir(exist_ok=False)
report=dict(video=s.name,frames=len(s.rgb),queries=len(s.query_txy),prediction_sha256=file_sha256(pred),metrics=rows,scope='Single completed DEVELOPMENT video. Coordinate/visibility swaps are posthoc diagnostics, not independently run ablations. No overall freeze decision.')
(out/'effects.json').write_text(json.dumps(report,indent=2)+'\n');x=np.arange(len(rows));fig,axes=plt.subplots(1,2,figsize=(13,5),constrained_layout=True)
for j,(key,label) in enumerate([('average_jaccard','AJ'),('occlusion_accuracy','Occlusion accuracy'),('average_pts_within_thresh','Position accuracy')]):axes[0].bar(x+(j-1)*.24,[100*v[key] for v in rows.values()],.24,label=label)
axes[0].set_xticks(x,['Raw','Raw XY\nFused vis','Fused XY\nRaw vis','Full']);axes[0].set(ylabel='Percent',ylim=(0,100),title='Saved-output swaps; not independent ablations');axes[0].legend(fontsize=8)
for j,(key,label) in enumerate([('false_hidden_visible_gt','Visible GT predicted hidden'),('false_visible_hidden_gt','Hidden GT predicted visible')]):axes[1].bar(np.arange(2)+(j-.5)*.35,[rows[k][key] for k in ['raw','full']],.35,label=label)
axes[1].set_xticks([0,1],['Raw','Full']);axes[1].set(ylabel='Point-frames',title='All evaluated point-frames; query frame excluded');axes[1].legend(fontsize=8)
for ax in axes:ax.grid(axis='y',alpha=.2)
fig.suptitle(exp.name+' | '+s.name+'\nOne whole robot DEV video; do not infer full development-set improvement');fig.savefig(out/'effects.png',dpi=150);print(json.dumps(report,indent=2))
