"""Prepare a deterministic full-video subset of RoboTAP evaluation shards.

Only split3/split4 (the upstream Track4World validation set) are considered.
Selection uses names and a seed, never model predictions or label quality.
"""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import pickle
import zipfile

import numpy as np

from .datasets import NumpyDatasetUnpickler, file_sha256, iter_robotap
from .geometry_worker import atomic_json


def prepare(archive, lock_path, output, fps, fps_description, videos=3, seed=20261029):
    if videos < 1 or not np.isfinite(fps) or fps <= 0 or not fps_description.strip():
        raise ValueError('Positive video count/FPS and explicit FPS provenance required')
    archive, lock_path = Path(archive).resolve(), Path(lock_path).resolve()
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    entry = json.loads(lock_path.read_text())['evaluation']
    if entry['dataset'] != 'robotap' or archive.stat().st_size != entry['bytes']:
        raise ValueError('Wrong or incomplete RoboTAP archive')
    digest = hashlib.md5()
    with archive.open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            digest.update(block)
    if digest.hexdigest() != entry['md5']:
        raise ValueError('RoboTAP archive checksum mismatch')
    # Parse records without unrestricted pickle globals or archive extraction.
    chosen = []
    source_shards = []
    with zipfile.ZipFile(archive) as bundle:
        for basename in ('robotap_split3.pkl', 'robotap_split4.pkl'):
            matches = [i for i in bundle.infolist() if PurePosixPath(i.filename).name == basename]
            if len(matches) != 1:
                raise ValueError('Missing or duplicate evaluation shard: ' + basename)
            info = matches[0]
            with bundle.open(info) as stream:
                records = NumpyDatasetUnpickler(stream).load()
                # Consume any remainder to trigger ZIP CRC validation.
                if stream.read(1):
                    raise ValueError('Unexpected bytes after dataset pickle')
            if not isinstance(records, dict) or not records or any(not isinstance(n, str) for n in records):
                raise ValueError('Expected named RoboTAP records')
            source_shards.append(dict(member=info.filename, crc32=info.CRC, bytes=info.file_size,
                                      video_count=len(records)))
            # Retain at most K from each shard; this preserves global top-K.
            ranked = sorted(records, key=lambda n: (hashlib.sha256(
                ('%d:%s:%s' % (seed, basename, n)).encode()).hexdigest(), n))[:videos]
            for name in ranked:
                rank = hashlib.sha256(('%d:%s:%s' % (seed, basename, name)).encode()).hexdigest()
                chosen.append((rank, basename, name, records[name]))
            del records
    total = sum(s['video_count'] for s in source_shards)
    if total < videos:
        raise ValueError('Requested more videos than available evaluation records')
    chosen = sorted(chosen, key=lambda r: r[:3])[:videos]
    if len({r[2] for r in chosen}) != len(chosen):
        raise ValueError('Duplicate selected video names across shards')
    output.mkdir(parents=True)
    atomic_json(output/'status.json', dict(state='preparing'))
    manifest = dict(schema_version=1, dataset='robotap', files=[], selected_videos=[],
                    source_archive_md5=entry['md5'], source_lock_sha256=file_sha256(lock_path),
                    source_shards=source_shards, seed=seed, fps_description=fps_description,
                    selection='lowest sha256(seed:shard:name); full frames and queryable tracks',
                    scope='selected evaluation subset, not full RoboTAP benchmark',
                    role='evaluation only; never calibration training supervision')
    try:
        for index, (_, shard, name, record) in enumerate(chosen):
            path = output / ('video_%06d.pkl' % index)
            with path.open('xb') as stream:
                pickle.dump({name: record}, stream, protocol=4)
            manifest['files'].append(dict(path=path.name, sha256=file_sha256(path), split='eval', fps=fps))
            manifest['selected_videos'].append(dict(name=name, source_shard=shard))
        atomic_json(output/'manifest.pending.json', manifest)
        summaries = []
        for sample in iter_robotap(output/'manifest.pending.json', 'eval'):
            summaries.append(dict(name=sample.name, frames=len(sample.rgb), queries=len(sample.query_txy)))
        (output/'manifest.pending.json').replace(output/'manifest.json')
        atomic_json(output/'status.json', dict(state='complete', videos=summaries))
    except BaseException as error:
        atomic_json(output/'status.json', dict(state='failed', error=type(error).__name__+': '+str(error)))
        raise
    return output/'manifest.json'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', default='data/downloads/robotap.zip')
    parser.add_argument('--lock', default='configs/datasets.lock.json')
    parser.add_argument('--output', required=True)
    parser.add_argument('--fps', type=float, required=True)
    parser.add_argument('--fps-description', required=True)
    parser.add_argument('--videos', type=int, default=3)
    parser.add_argument('--seed', type=int, default=20261029)
    args = parser.parse_args()
    print(prepare(args.archive, args.lock, args.output, args.fps,
                  args.fps_description, args.videos, args.seed))


if __name__ == '__main__':
    main()
