"""Produce explanatory evidence for each published video without restarting work.

Uses live_replay_audits, separate from the main controller's final audit paths.
Only published, hash-bound videos are processed. GT is used by the posthoc
effect diagnostic, never supplied to the running inference pipeline.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment', required=True, type=Path)
    args = parser.parse_args()
    exp = args.experiment.resolve()
    root = Path(__file__).resolve().parents[1]
    out = exp/'live_explanation_watch'
    out.mkdir(exist_ok=False)
    status = json.loads((exp/'status.json').read_text())
    pid = int(status['pid'])
    scripts = [exp/'tools/audit_calibration_recording.py',
               root/'scripts/diagnose_video_effects.py',
               root/'scripts/render_calibration_evidence.py']
    hashes = {str(p): sha(p) for p in scripts}
    hashes[str(Path(__file__).resolve())] = sha(Path(__file__).resolve())
    (out/'source_manifest.json').write_text(json.dumps(hashes, indent=2)+'\n')
    report = dict(state='running', watched_pid=pid, watcher_pid=os.getpid(), videos=[])

    def write():
        temporary = out/'status.pending.json'
        temporary.write_text(json.dumps(report, indent=2)+'\n')
        temporary.replace(out/'status.json')

    def run(label, script, arguments):
        for filename, digest in hashes.items():
            assert sha(Path(filename)) == digest, 'Diagnostic code changed: '+filename
        with (out/(label+'.log')).open('x') as log:
            subprocess.run([sys.executable, str(script), *arguments], cwd=root,
                env={**os.environ, 'OMP_NUM_THREADS':'2', 'OPENBLAS_NUM_THREADS':'1',
                     'MPLCONFIGDIR':'/tmp/tracking-mpl'},
                stdout=log, stderr=subprocess.STDOUT, check=True)

    write()
    try:
        while True:
            progress = exp/'development/progress.json'
            entries = json.loads(progress.read_text())['videos'] if progress.exists() else []
            for i, entry in enumerate(entries):
                if i < len(report['videos']):
                    assert report['videos'][i]['sha256'] == entry['sha256']
                    continue
                track = exp/'development'/entry['path']
                assert sha(track) == entry['sha256']
                audit = exp/'live_replay_audits'/f'video_{i:06d}.json'
                common = ['--experiment', str(exp)]
                run(f'{i}_replay', scripts[0], common+['--tracks', str(track), '--output', str(audit)])
                run(f'{i}_effects', scripts[1], common+['--video', str(i)])
                run(f'{i}_evidence', scripts[2], common+['--video', str(i), '--audit', str(audit)])
                report['videos'].append(dict(index=i, name=entry['name'], sha256=entry['sha256']))
                write()
            current = json.loads((exp/'status.json').read_text())
            if current['state'] == 'complete':
                expected = json.loads((exp/'development/predictions.json').read_text())['videos']
                if len(report['videos']) != len(expected):
                    continue
                report['state'] = 'complete'
                write()
                return
            if current['state'] in ['failed', 'interrupted', 'not_started']:
                raise RuntimeError('Main workflow terminal: '+current['state'])
            command = Path(f'/proc/{pid}/cmdline')
            if not command.exists() or str(exp/'run.py').encode() not in command.read_bytes():
                if json.loads((exp/'status.json').read_text())['state'] == 'complete':
                    continue
                raise RuntimeError('Controller absent or identity changed; no restart attempted')
            time.sleep(20)
    except BaseException as error:
        report.update(state='failed', error=repr(error))
        write()
        raise


if __name__ == '__main__':
    main()
