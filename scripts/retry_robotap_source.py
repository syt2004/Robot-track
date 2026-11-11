"""Back off after confirmed source failures; never duplicate a live downloader."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True)
    p.add_argument('--workers',type=int,default=4)
    p.add_argument('--initial-delay',type=int,default=300)
    a=p.parse_args()
    if not 1<=a.workers<=16 or a.initial_delay<0:p.error('Invalid workers or delay')
    root=Path(__file__).resolve().parents[1]
    out=Path(a.output).resolve();out.mkdir(parents=True,exist_ok=False)
    def status(**values):
        temporary=out/'status.pending.json';temporary.write_text(json.dumps(values,indent=2));temporary.replace(out/'status.json')
        print(json.dumps(values),flush=True)
    for attempt in range(1,13):
        delay=a.initial_delay if attempt==1 else min(300*attempt,900)
        status(state='backoff',attempt=attempt,delay_seconds=delay)
        time.sleep(delay)
        log_path=out/('attempt-%02d.log'%attempt)
        with log_path.open('x') as log:
            process=subprocess.Popen([sys.executable,'-u',str(root/'scripts/download_robotap_prefix.py'),'--workers',str(a.workers)],
                                     cwd=root,stdout=log,stderr=subprocess.STDOUT)
            status(state='downloading',attempt=attempt,pid=process.pid,log=str(log_path))
            try:code=process.wait()
            except BaseException:
                process.terminate();process.wait();status(state='interrupted',attempt=attempt);raise
        if code==0:
            status(state='verified_member',attempt=attempt,whole_archive_verified=False);return
        status(state='attempt_failed',attempt=attempt,returncode=code)
        # Retries occur only after the owned child has actually exited.
    status(state='failed',attempts=12)
    raise SystemExit(1)


if __name__=='__main__':main()
