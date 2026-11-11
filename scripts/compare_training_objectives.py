"""Compare matched completed CPU TRAIN-only probes; no development claims."""
from pathlib import Path
import json,sys,hashlib
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path(__file__).resolve().parents[1];runs=[root/'runs/objective-control-cpu-001',root/'runs/objective-probe-cpu-002'];audits=[];states=[]
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
for p in runs:
 assert json.loads((p/'status.json').read_text())['state']=='complete'
 a=json.loads((p/'fit_audit/audit.json').read_text());s=torch.load(p/'training/calibrator.pt',map_location='cpu',weights_only=True)
 assert a['checkpoint_sha256']==sha(p/'training/calibrator.pt')
 assert a['manifest_sha256']==sha(p/'training_manifest.json')==s['manifest_sha256']
 assert s['epoch']==20 and s['resumed_from_sha256'] is None
 assert sum(r['labeled_point_frames'] for r in a['rows'].values())==96263
 audits.append(a);states.append(s)
assert states[0]['training_config']==states[1]['training_config'] and states[0]['training_config']['device_type']=='cpu'
assert audits[0]['manifest_sha256']==audits[1]['manifest_sha256']
sm=[json.loads((p/'source_manifest.json').read_text()) for p in runs];assert sm[0].keys()==sm[1].keys()
diffs=[k for k in sm[0] if sm[0][k]!=sm[1][k]];assert diffs==['geoverify/calibration.py'],diffs
rows={}
for k,r in audits[0]['rows'].items():
 q=audits[1]['rows'][k]
 assert r['labeled_point_frames']==q['labeled_point_frames'] and r['visible_point_frames']==q['visible_point_frames']
 assert r['raw_error_px_sum']==q['raw_error_px_sum'] and r['raw_within_threshold_counts']==q['raw_within_threshold_counts']
 rows[k]={name:{m:v[m] for m in ['fused_mean_error_pixels','fused_visibility_accuracy','fused_average_position_threshold_accuracy']} for name,v in [('hard_rank',r),('expected_regret',q)]}
out=root/'runs/objective-cpu-comparison-001';out.mkdir(exist_ok=False)
report=dict(state='complete',scope='Matched TRAIN cache fit only; not full video inference, held-out evaluation, or acceptance. Loss values across objectives are not comparable.',inputs={str(p):sha(p/'fit_audit/audit.json') for p in runs},source_changed_files=diffs,training_config=states[0]['training_config'],rows=rows)
(out/'comparison.json').write_text(json.dumps(report,indent=2)+'\n')
keys=[k for k in rows if not k.startswith('4541')];assert len(keys)==3
fig,axs=plt.subplots(1,3,figsize=(14,4),constrained_layout=True)
for ax,(metric,title,scale) in zip(axs,[('mean_error_pixels','Visible point error (px; lower better)',1),('average_position_threshold_accuracy','Position threshold accuracy (%)',100),('visibility_accuracy','Visibility accuracy (%)',100)]):
 for j,(label,data,stem) in enumerate([('Raw',audits[0],'raw'),('Hard rank',audits[0],'fused'),('Expected regret',audits[1],'fused')]):
  ax.bar([i+(j-1)*.23 for i in range(3)],[data['rows'][k][stem+'_'+metric]*scale for k in keys],.23,label=label)
 ax.set_xticks(range(3),['plush','cables','gear']);ax.set_title(title,fontsize=10);ax.grid(axis='y',alpha=.2);ax.legend(fontsize=8)
fig.suptitle('Matched CPU TRAIN-only objective comparison | 20 epochs, seed 0, identical caches\nFit diagnostic only; no DEV/report data or thresholds selected')
fig.savefig(out/'robot_comparison.png',dpi=150)
print(json.dumps(report,indent=2))
