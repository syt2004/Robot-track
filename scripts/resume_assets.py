"""Resume locked visual assets with independent logs and bounded retries.

Each child exclusively locks its partial file. Retry only follows a confirmed
nonzero child exit; a quiet log is never treated as process death.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time


def run(output,workers=16,retries=4,robotap_prefix=False,metric_only=False,track4world_only=False):
    root=Path(__file__).resolve().parents[1]
    out=Path(output).resolve();out.mkdir(parents=True,exist_ok=False)
    assets=json.loads((root/'configs/assets.lock.json').read_text())['downloads']
    robot=json.loads((root/'configs/datasets.lock.json').read_text())['evaluation']
    jobs={name:dict(entry=assets[name],algorithm='sha256',workers=workers) for name in ('track4world_da3','da3_nested')}
    if metric_only:
        selection=json.loads((root/'configs/da3.metric.lock.json').read_text())
        del jobs['da3_nested']
        jobs['da3_metric']=dict(entry=dict(path='weights/da3nested-metric/metric.payload',bytes=selection['payload_bytes'],
            url=selection['source_url'],source_sha256=selection['source_sha256']),algorithm='selected_tensors',workers=16)
    jobs['robotap']=dict(entry=dict(path='data/downloads/robotap.zip',bytes=robot['bytes'],md5=robot['md5'],
        url='https://storage.googleapis.com/download/storage/v1/b/dm-tapnet/o/robotap%2Frobotap.zip?alt=media&generation=1693927735577112'),algorithm='md5',workers=4)
    if robotap_prefix:
        index=json.loads((root/'configs/robotap.zip-index.json').read_text())
        entry=next(e for e in index['entries'] if e['name']=='robotap/robotap_split4.pkl')
        boundary=min(e['offset'] for e in index['entries'] if e['offset']>entry['offset'])
        del jobs['robotap']
        jobs['robotap_split4']=dict(entry=dict(path='data/downloads/robotap-split4-prefix.bin',bytes=boundary,
            url=index['source'],member=entry['name'],crc32=entry['crc']),algorithm='member_crc32',workers=4)
    if track4world_only:jobs={'track4world_da3':jobs['track4world_da3']}
    active={}
    for job in jobs.values():job.update(attempts=0,state='pending',pid=None)
    def save():
        for job in jobs.values():
            p=root/job['entry']['path'];part=p.with_name(p.name+'.part')
            job['actual_bytes']=p.stat().st_size if p.exists() else part.stat().st_size if part.exists() else 0
        temp=out/'status.pending.json';temp.write_text(json.dumps(jobs,indent=2));temp.replace(out/'status.json')
    def launch(name):
        job=jobs[name];e=job['entry'];job['attempts']+=1
        log=out/('%s-attempt-%d.log'%(name,job['attempts']))
        if job['algorithm']=='member_crc32':
            command=[sys.executable,'-u',str(root/'scripts/download_robotap_prefix.py')]
        elif job['algorithm']=='selected_tensors':
            command=[sys.executable,'-u',str(root/'scripts/download_da3_metric.py')]
        else:
            command=[sys.executable,'-u',str(root/'scripts/download_parallel.py'),'--url',e['url'],
                     '--output',str(root/e['path']),'--'+job['algorithm'],e[job['algorithm']],
                     '--workers',str(job['workers']),'--chunk-bytes',str(1024*1024)]
        stream=log.open('xb')
        process=subprocess.Popen(command,cwd=root,stdout=stream,stderr=subprocess.STDOUT)
        active[name]=(process,stream)
        job.update(state='running',pid=process.pid,log=str(log),command=command)
        print(name,'attempt',job['attempts'],'pid',process.pid,flush=True)
    try:
        for name in jobs:launch(name)
        save();last=time.monotonic()
        while active:
            for name,(process,stream) in list(active.items()):
                code=process.poll()
                if code is None:continue
                stream.close();del active[name]
                jobs[name].update(state='verified' if code==0 else 'failed',returncode=code)
                print(name,jobs[name]['state'],'exit',code,flush=True)
                if code!=0 and jobs[name]['attempts']<retries:launch(name)
                save()
            if time.monotonic()-last>=30:
                save();print({k:(j['state'],j['actual_bytes'],j['entry']['bytes']) for k,j in jobs.items()},flush=True);last=time.monotonic()
            time.sleep(1)
    except BaseException:
        for name,(process,stream) in active.items():
            process.terminate()
            try:process.wait(timeout=10)
            except subprocess.TimeoutExpired:process.kill();process.wait()
            stream.close();jobs[name]['state']='interrupted'
        save();raise
    return all(j['state']=='verified' for j in jobs.values())


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True);parser.add_argument('--workers',type=int,default=16)
    parser.add_argument('--retries',type=int,default=4)
    parser.add_argument('--robotap-prefix',action='store_true',help='Only split4 ZIP prefix, verified by member CRC; not full archive MD5')
    parser.add_argument('--metric-only',action='store_true',help='Exact DA3 metric tensors only; Track4World supplies the entire other branch')
    parser.add_argument('--track4world-only',action='store_true',help='Resume only full Track4World checkpoint; leave other download owners untouched')
    a=parser.parse_args()
    if not 1<=a.workers<=64 or not 1<=a.retries<=10:parser.error('Invalid bounded workers/retries')
    raise SystemExit(0 if run(a.output,a.workers,a.retries,a.robotap_prefix,a.metric_only,a.track4world_only) else 1)
