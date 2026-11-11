"""Posthoc robot DEV geometry-weight and candidate-gain diagnostic, no fitting."""
import argparse,json,sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root))
from geoverify.datasets import iter_evaluation,file_sha256
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--experiment',type=Path,required=True);p.add_argument('--video',type=int,required=True);a=p.parse_args();exp=a.experiment.resolve()
manifest=root/'runs/robot-development-001/manifest.json';progress=json.loads((exp/'development/progress.json').read_text());entry=progress['videos'][a.video]
assert progress['dataset_manifest_sha256']==file_sha256(manifest)
s=next(x for i,x in enumerate(iter_evaluation(manifest,'dev')) if i==a.video);assert s.name==entry['name']
path=exp/'development'/entry['path'];assert file_sha256(path)==entry['sha256']
z=dict(np.load(path,allow_pickle=False));np.testing.assert_array_equal(z['query_txy'],s.query_txy)
mask=(np.arange(len(s.rgb))[:,None]>s.query_txy[None,:,0]) & s.target_visible & z['processed'].astype(bool)
scale=np.array([256/s.rgb.shape[2],256/s.rgb.shape[1]])
candidate=np.linalg.norm((z['candidates']-s.target_xy[:,:,None,:])*scale,axis=-1)
valid=z['candidate_valid'].astype(bool);raw=candidate[...,0];fused=np.linalg.norm((z['xy']-s.target_xy)*scale,axis=-1)
weight=z['fusion_weights'][...,1:3].sum(-1);rows=[]
for k in range(4):
 use=mask & valid[...,k];err=candidate[...,k][use];ref=raw[use]
 rows.append(dict(candidate=k+1,rows=int(use.sum()),raw_mean=float(ref.mean()) if len(ref) else None,candidate_mean=float(err.mean()) if len(err) else None,better_over1px=float((err<ref-1).mean()) if len(err) else None,worse_over1px=float((err>ref+1).mean()) if len(err) else None))
bins=[]
for lo,hi in zip([0,.05,.1,.2,.4,.6],[.05,.1,.2,.4,.6,1.000001]):
 use=mask & (weight>=lo) & (weight<hi);n=int(use.sum())
 bins.append(dict(lo=lo,hi=min(hi,1.),rows=n,mean_gain_px=float((raw[use]-fused[use]).mean()) if n else None,help_fraction=float((fused[use]<raw[use]-1).mean()) if n else None,hurt_fraction=float((fused[use]>raw[use]+1).mean()) if n else None))
out=path.parent/'geometry_gain';out.mkdir(exist_ok=False)
r=dict(video=s.name,prediction_sha256=file_sha256(path),script_sha256=file_sha256(Path(__file__)),candidate_rows=rows,weight_bins=bins,scope='Posthoc labeled visible robot DEV point-frames, query frame excluded. Each candidate compared on its own valid rows; counts differ. Weight bins fixed before this analysis. No fitted threshold; not a causal or independently rerun geometry ablation.')
(out/'diagnostic.json').write_text(json.dumps(r,indent=2,allow_nan=False)+'\n')
fig,ax=plt.subplots(1,2,figsize=(13,5),constrained_layout=True)
x=np.arange(4);ax[0].bar(x-.18,[v['raw_mean'] or 0 for v in rows],.36,label='Raw on same valid rows');ax[0].bar(x+.18,[v['candidate_mean'] or 0 for v in rows],.36,label='Candidate');ax[0].set_xticks(x,[f"C{i+1}\nn={v['rows']}" for i,v in enumerate(rows)]);ax[0].set(ylabel='Mean visible error, 256-scaled pixels',title='Candidate quality (different validity masks)');ax[0].legend(fontsize=8)
x=np.arange(len(bins));ax[1].bar(x,[v['mean_gain_px'] if v['rows'] else np.nan for v in bins]);ax[1].axhline(0,color='black',lw=1);ax[1].set_xticks(x,[f"{v['lo']:.2f}-{v['hi']:.2f}\nn={v['rows']}" for v in bins],rotation=30);ax[1].set(ylabel='Raw error minus fused error (positive helps)',title='Saved prediction gain versus geometry weight')
for axis in ax:axis.grid(axis='y',alpha=.2)
fig.suptitle(s.name+' | robot DEV diagnostic\nGT used only for analysis; not independent ablation');fig.savefig(out/'diagnostic.png',dpi=150);plt.close(fig);print(json.dumps(r,indent=2))
