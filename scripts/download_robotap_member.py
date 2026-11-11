"""Download one pinned RoboTAP ZIP member by bounded ranges, preserving resume."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import io
import json
import os
from pathlib import Path
import sys

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))
from scripts.download_parallel import fetch_range
from geoverify.robotap_prefix import member_header, extract_member
from geoverify.datasets import file_sha256
from geoverify.geometry_worker import atomic_json


def download_member(index_path, split, output_dir, status_path, workers=4, fetch=fetch_range):
    index = json.loads(Path(index_path).read_text())
    entry = next(e for e in index['entries'] if e['name'] == f'robotap/robotap_split{split}.pkl')
    local_entry = {**entry, 'offset': 0}
    offset = entry['offset']
    header, _ = fetch(index['source'], offset, offset + 4095, index['bytes'])
    data_start = member_header(io.BytesIO(header), local_entry)
    length = data_start + entry['compressed']
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    packed = output_dir / f'robotap-split{split}-member.bin'
    part = packed.with_suffix(packed.suffix + '.part')
    member = output_dir / f'robotap_split{split}.pkl'
    state = dict(state='running', stage='download', pid=os.getpid(), split=split,
                 requested_member_bytes=length, archive_bytes=index['bytes'],
                 source_index_sha256=file_sha256(index_path))
    atomic_json(status_path, state)
    try:
        if not packed.exists():
            with part.open('ab+') as stream:
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                stream.seek(0, 2)
                position = stream.tell()
                assert position <= length, 'Oversized partial member'
                if position >= data_start:
                    stream.seek(0)
                    assert stream.read(data_start) == header[:data_start], 'Partial header mismatch'
                    stream.seek(position)
                with ThreadPoolExecutor(max_workers=workers) as pool:
                    pending, next_start = {}, position
                    def fill():
                        nonlocal next_start
                        while len(pending) < workers and next_start < length:
                            end = min(next_start + 1048576, length) - 1
                            pending[next_start] = pool.submit(fetch, index['source'], offset + next_start, offset + end, index['bytes'])
                            next_start = end + 1
                    fill()
                    while pending:
                        data, _ = pending.pop(position).result()
                        stream.write(data)
                        stream.flush()
                        os.fsync(stream.fileno())
                        position += len(data)
                        state['downloaded_bytes'] = position
                        atomic_json(status_path, state)
                        fill()
                assert position == length
            part.replace(packed)
        assert packed.stat().st_size == length
        state.update(stage='extract_verify', downloaded_bytes=length)
        atomic_json(status_path, state)
        record = extract_member(packed, local_entry, member)
        record.update(source=index['source'], source_index_sha256=file_sha256(index_path),
                      archive_member_offset=offset, archive_data_offset=offset + data_start,
                      downloaded_range=[offset, offset + length - 1],
                      downloaded_member_sha256=file_sha256(packed), downloaded_member_bytes=length,
                      note='CRC32 and declared size verified against pinned ZIP index; local SHA256 recorded. Whole archive MD5 not verified.')
        atomic_json(member.with_suffix('.provenance.json'), record)
        state.update(state='complete', stage='finished', member=str(member), member_sha256=record['sha256'])
        atomic_json(status_path, state)
        print(json.dumps(state), flush=True)
        return record
    except Exception as error:
        state.update(state='failed', error=type(error).__name__)
        atomic_json(status_path, state)
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--split', type=int, choices=range(5), required=True)
    parser.add_argument('--status', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=4, choices=range(1, 17))
    args = parser.parse_args()
    download_member(root / 'configs/robotap.zip-index.json', args.split,
                    root / 'data/downloads', args.status, args.workers)
