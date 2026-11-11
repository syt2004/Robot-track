"""Explain actual saved robot-video calibration inputs and decisions without GT."""
from pathlib import Path
import argparse,json,sys
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--experiment',required=True);p.add_argument('--video',type=int,required=True);p.add_argument('--audit',type=Path);args=p.parse_args();exp=Path(args.experiment).resolve();video=exp/'development'/f'video_{args.video:06d}';out=video/'calibration_evidence'
audit_path=args.audit or exp/'calibration_replay_audits'/f'video_{args.video:06d}.json'
audit=json.loads(audit_path.read_text());assert audit['state']=='passed' and audit['prediction_manifest_bound']
sys.path.insert(0,str(exp/'source'))
import geoverify.calibration as cal
from geoverify.datasets import file_sha256
config=json.loads((exp/'inference.json').read_text());state=torch.load(config['calibrator_checkpoint'],map_location='cpu',weights_only=True);a=dict(np.load(video/'tracks.npz'));mask=a['processed'].astype(bool);valid=a['candidate_valid'].astype(bool);features=a['calibration_features'].copy()
assert audit['tracks_sha256']==file_sha256(video/'tracks.npz'),'Replay audit belongs to different predictions'
assert audit['checkpoint_sha256']==file_sha256(config['calibrator_checkpoint']),'Replay audit belongs to different checkpoint'
out.mkdir(exist_ok=False)
if hasattr(cal,'canonical_features'):features=cal.canonical_features(torch.from_numpy(features),torch.from_numpy(valid)).numpy()
z=(features-state['model']['feature_mean'].numpy())/state['model']['feature_std'].numpy()
frame=np.arange(len(mask));time=a['timestamps']-a['timestamps'][0];raw=a['candidate_visibility'][...,0];pred=a['visibility'];geometry=valid[...,1:3].any(-1);weights=a['fusion_weights'][...,1:3].sum(-1)
def mean(x):return np.array([np.mean(x[t][mask[t]]) for t in frame])
def med(x):return np.array([np.median(x[t][mask[t]]) for t in frame])
rawage=a.get('geometry_wall_age',a['geometry_age']).copy();age_med=np.array([np.median(row[np.isfinite(row)]) if np.isfinite(row).any() else np.nan for row in rawage]);norm95=np.array([np.quantile(np.abs(z[t][valid[t]]),.95,axis=0) for t in frame]);top=np.argsort(np.quantile(norm95,.95,axis=0))[-3:][::-1]
fig,axes=plt.subplots(2,2,figsize=(13,8),constrained_layout=True)
axes[0,0].plot(time,mean(raw),label='Current Track-On probability');axes[0,0].plot(time,mean(pred),label='Fused visibility probability');axes[0,0].set(title='Mean visibility over active queries',ylabel='Probability',ylim=(0,1));axes[0,0].legend(fontsize=8)
axes[0,1].plot(time,mean(geometry),label='Queries with usable geometry');axes[0,1].plot(time,mean(weights),label='Mean geometry fusion weight');axes[0,1].plot(time,mean(a['geometry_expired']),ls='--',label='Queries with expired evidence');axes[0,1].set(title='Available evidence and actual contribution',ylabel='Fraction / weight',ylim=(0,1));axes[0,1].legend(fontsize=8)
axes[1,0].plot(time,age_med,label='Recorded wall-clock age');axes[1,0].plot(time,med(features[...,0,11]),label='Model input age (missing = -1)');axes[1,0].axhline(config['pipeline']['max_geometry_age'],color='red',ls=':',label='Candidate expiry budget');axes[1,0].set(title='Age diagnostics versus model input',ylabel='Seconds');axes[1,0].legend(fontsize=8)
for k in top:axes[1,1].plot(time,norm95[:,k],label=cal.FEATURE_NAMES[k])
axes[1,1].axhline(5,color='grey',ls=':',label='5 training standard deviations');axes[1,1].set(title='95th percentile |normalized feature|; top three',ylabel='Training standard deviations');axes[1,1].legend(fontsize=8)
for ax in axes.flat:ax.set_xlabel('Capture time (seconds)');ax.grid(alpha=.2)
fig.suptitle(exp.name+f' / robot DEV video {args.video}: saved model evidence\nNo GT used in these plots; input/output CPU replay verified; not an accuracy claim')
fig.savefig(out/'evidence.png',dpi=150);plt.close(fig)
summary=dict(processed_point_frames=int(mask.sum()),fraction_visibility_class_changed=float(((raw>=.5)!=(pred>=.5))[mask].mean()),geometry_available_fraction=float(geometry[mask].mean()),mean_geometry_weight=float(weights[mask].mean()),largest_normalized_features=[cal.FEATURE_NAMES[k] for k in top],audit=str(audit_path.resolve()),scope='Full saved video, active queries, no GT; illustrative aggregates do not establish accuracy.')
(out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))
