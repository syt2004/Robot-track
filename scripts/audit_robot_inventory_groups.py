"""Conservative cross-member grouping; does not assign train/test splits."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path


def group_rows(rows, frame_hashes, train_prefixes, held_prefixes):
    ids = [r['id'] for r in rows]
    assert len(ids) == len(set(ids)), 'Duplicate inventory identity'
    parent = {k: k for k in ids}
    def find(k):
        while parent[k] != k:
            parent[k] = parent[parent[k]]
            k = parent[k]
        return k
    def join(a, b):
        parent[find(b)] = find(a)
    prefixes, frames = {}, {}
    overlaps = defaultdict(int)
    for row in rows:
        key, prefix = row['id'], row['prefix_hint']
        hashes = frame_hashes[key]
        assert len(hashes) == row['frames']
        if prefix in prefixes:
            join(key, prefixes[prefix])
        else:
            prefixes[prefix] = key
        for digest in set(hashes):
            if digest in frames:
                other = frames[digest]
                join(key, other)
                overlaps[tuple(sorted([key, other]))] += 1
            else:
                frames[digest] = key
    groups = defaultdict(list)
    for row in rows:
        groups[find(row['id'])].append(row)
    result = []
    for members in groups.values():
        prefixes = {r['prefix_hint'] for r in members}
        known_train, known_held = bool(prefixes & train_prefixes), bool(prefixes & held_prefixes)
        result.append(dict(members=members, touches_known_train=known_train,
                           touches_previously_exposed_held=known_held,
                           available_for_new_split_design=not (known_train or known_held)))
    return dict(groups=result, overlap_links=[dict(left=a, right=b, shared_exact_frames=n) for (a, b), n in sorted(overlaps.items())],
                known_role_conflicts=sum(g['touches_known_train'] and g['touches_previously_exposed_held'] for g in result))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inventory', action='append', type=Path, required=True)
    p.add_argument('--existing-training-manifest', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    rows, hashes, inputs = [], {}, {}
    for directory in args.inventory:
        assert json.loads((directory / 'status.json').read_text())['state'] == 'complete'
        inv = json.loads((directory / 'inventory.json').read_text())
        frames = json.loads((directory / 'decoded_frame_hashes.json').read_text())
        for row in inv['videos']:
            key = inv['source_sha256'] + ':' + row['name']
            rows.append({**row, 'id': key, 'inventory': str(directory.resolve())})
            hashes[key] = frames[row['name']]
        for name in ['inventory.json', 'decoded_frame_hashes.json']:
            path = directory / name
            inputs[str(path.resolve())] = hashlib.sha256(path.read_bytes()).hexdigest()
    old = json.loads(args.existing_training_manifest.read_text())
    inputs[str(args.existing_training_manifest.resolve())] = hashlib.sha256(args.existing_training_manifest.read_bytes()).hexdigest()
    report = group_rows(rows, hashes, {r['episode_prefix'] for r in old['selected_videos']}, set(old['excluded_episode_prefixes']))
    report.update(state='complete_exact_group_audit', videos=len(rows), groups_count=len(report['groups']),
                  available_group_count=sum(g['available_for_new_split_design'] for g in report['groups']),
                  inputs_sha256=inputs,
                  scope='Conservative unions of name prefix and any exact RGB overlap. Not verified capture sessions, not a near-duplicate audit, no split assigned. Known held exposure inherited from prior training manifest exclusions.')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as f:
        json.dump(report, f, indent=2)
    print(json.dumps({k: report[k] for k in ['videos', 'groups_count', 'available_group_count', 'known_role_conflicts']}))


if __name__ == '__main__':
    main()
