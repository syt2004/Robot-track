"""Verify completed training, source snapshot, data and promoted model bindings."""
import argparse,json,hashlib,math
from pathlib import Path
import torch

def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
 return h.hexdigest()
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--experiment',required=True);p.add_argument('--output',required=True);args=p.parse_args();exp=Path(args.experiment).resolve();output=Path(args.output)
assert not output.exists()
status=json.loads((exp/'status.json').read_text());train=next(v for v in status['stages'] if v['name']=='train');promote=next(v for v in status['stages'] if v['name']=='promote');assert train['state']==promote['state']=='complete'
for name,h in json.loads((exp/'source_manifest.json').read_text()).items():assert sha(exp/'source'/name)==h,name
manifest=exp/'training_manifest.json';data=json.loads(manifest.read_text());sources=sorted({s['source_sha256'] for s in data['shards']});assert all(s['split']=='train' for s in data['shards'])
for s in data['shards']:assert sha((manifest.parent/s['path']).resolve())==s['sha256']
state=torch.load(exp/'training/calibrator.pt',map_location='cpu',weights_only=True);config=json.loads((exp/'inference.json').read_text());history=json.loads((exp/'training/training_history.json').read_text());command=train['command'];value=lambda flag:command[command.index(flag)+1]
assert state['epoch']==int(value('--epochs'))==len(history);assert history==state['history'];assert [v['epoch'] for v in history]==list(range(1,len(history)+1));assert all(math.isfinite(v['loss']) and v['steps']>0 for v in history)
assert state['seed']==int(value('--seed'));assert state['training_config']['lr']==float(value('--lr'))
batchflag='--batch-points' if '--batch-points' in command else '--batch-frames';batchkey=batchflag[2:].replace('-','_');assert state['training_config'][batchkey]==int(value(batchflag))
assert state['resumed_from_sha256'] is None and '--resume' not in command
assert state['manifest_sha256']==sha(manifest);assert state['training_source_sha256']==sources
assert state['baseline_provenance']==config['baseline_provenance']==data['baseline_provenance'];assert state['head_architecture']==config['head_architecture'];assert state['geometry_intrinsics_protocol']==config['geometry_intrinsics_protocol']==data['geometry_intrinsics_protocol']
assert state.get('evidence_age_protocol')==config.get('evidence_age_protocol')==data.get('evidence_age_protocol')
checkpoint=(exp/'training/calibrator.pt').resolve();assert Path(config['calibrator_checkpoint']).resolve()==checkpoint;assert config['calibrator_sha256']==sha(checkpoint)
assert all(bool(torch.isfinite(v).all()) for v in state['model'].values());assert (state['model']['feature_std']>0).all()
report=dict(state='passed',epochs=state['epoch'],initial_loss=history[0]['loss'],final_loss=history[-1]['loss'],checkpoint_sha256=sha(checkpoint),training_manifest_sha256=sha(manifest),head_architecture=state['head_architecture'],training_config=state['training_config'],training_sources=sources,steps_per_epoch=[h['steps'] for h in history],checks=['frozen source hashes','every training shard hash','complete training history','no resumed checkpoint','optimizer arguments','finite model and normalization','promoted inference checkpoint binding'],limitation='Development training may reuse earlier caches; this is not final from-scratch reproduction or accuracy evidence.')
output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
