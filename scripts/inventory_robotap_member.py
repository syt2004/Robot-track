"""Inventory a verified RoboTAP member without model evaluation or split assignment."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import numpy as np

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))
from geoverify.datasets import NumpyDatasetUnpickler, file_sha256, robotap_record
from geoverify.geometry_worker import atomic_json

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--member', required=True, type=Path)
p.add_argument('--output', required=True, type=Path)
args = p.parse_args()
args.output.mkdir(parents=True, exist_ok=False)
state = dict(state='running', pid=os.getpid(), stage='verify_source')
atomic_json(args.output / 'status.json', state)
try:
    provenance = json.loads(args.member.with_suffix('.provenance.json').read_text())
    assert file_sha256(args.member) == provenance['sha256']
    with args.member.open('rb') as stream:
        records = NumpyDatasetUnpickler(stream).load()
    assert isinstance(records, dict) and records
    rows, signatures = [], {}
    for name in sorted(records):
        sample = robotap_record(name, 'unassigned', records[name], 24., provenance['sha256'])
        hashes = [hashlib.sha256(frame.tobytes()).hexdigest() for frame in sample.rgb]
        # Names are only conservative grouping hints, not verified capture sessions.
        rows.append(dict(name=name, prefix_hint=name.split('-', 1)[0], frames=len(sample.rgb),
                         height=sample.rgb.shape[1], width=sample.rgb.shape[2],
                         queries=len(sample.query_txy), query_cohorts=len(np.unique(sample.query_txy[:, 0])),
                         decoded_unique_frames=len(set(hashes)),
                         ordered_rgb_sequence_sha256=hashlib.sha256(''.join(hashes).encode()).hexdigest()))
        signatures[name] = hashes
        state.update(stage='inventory', completed_videos=len(rows), total_videos=len(records), video=name)
        atomic_json(args.output / 'status.json', state)
    atomic_json(args.output / 'inventory.json', dict(state='complete', source_sha256=provenance['sha256'],
                videos=rows, total_videos=len(rows), prefix_hints=len({r['prefix_hint'] for r in rows}),
                scope='Metadata and exact decoded RGB hashes only; no predictions, performance selection, split assignment or unseen-session guarantee. Near duplicates still require screening.'))
    atomic_json(args.output / 'decoded_frame_hashes.json', signatures)
    state.update(state='complete', stage='finished')
    atomic_json(args.output / 'status.json', state)
    print(json.dumps(dict(videos=len(rows), frames=sum(r['frames'] for r in rows), prefix_hints=len({r['prefix_hint'] for r in rows}))), flush=True)
except Exception as error:
    state.update(state='failed', error=repr(error))
    atomic_json(args.output / 'status.json', state)
    raise
