"""Audit completed video records while an existing experiment continues."""
from pathlib import Path
import sys,json,time,subprocess,os,argparse
parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--experiment',required=True);args=parser.parse_args()
exp=Path(args.experiment).resolve();out=exp/'calibration_replay_audits';out.mkdir(exist_ok=False)
launch=json.loads((exp/'launch.json').read_text());pid=launch['pid'];script=launch['script'];root=Path(__file__).resolve().parents[1]
def write(state,**extra):
 report=dict(state=state,watched_pid=pid,completed=completed,**extra);temp=out/'status.pending.json';temp.write_text(json.dumps(report,indent=2)+'\n');temp.replace(out/'status.json')
completed=[];write('running')
try:
 while True:
  progress=exp/'development/progress.json'
  if progress.exists():
   for i,entry in enumerate(json.loads(progress.read_text())['videos']):
    if i in completed:continue
    target=out/f'video_{i:06d}.json'
    command=[sys.executable,str(root/'scripts/audit_calibration_recording.py'),'--experiment',str(exp),'--tracks',str(progress.parent/entry['path']),'--output',str(target)]
    with (out/f'video_{i:06d}.log').open('x') as log:subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True,env={**os.environ,'OMP_NUM_THREADS':'1'})
    completed.append(i);write('running')
  status=json.loads((exp/'status.json').read_text())
  if status['state']=='complete':
   expected=len(json.loads((exp/'development/predictions.json').read_text())['videos'])
   if len(completed)!=expected:continue
   write('complete');break
  if status['state'] in ['failed','interrupted']:raise RuntimeError('Experiment ended '+status['state'])
  process=Path(f'/proc/{pid}/cmdline')
  if not process.exists() or script.encode() not in process.read_bytes():
   # Re-read terminal state to avoid a race at successful workflow exit.
   if json.loads((exp/'status.json').read_text())['state']=='complete':continue
   raise RuntimeError('Watched experiment process absent or identity changed; no restart attempted')
  time.sleep(20)
except BaseException as e:
 write('failed',error=repr(e));raise
