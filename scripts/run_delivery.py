from pathlib import Path
import argparse
import subprocess

R = Path(__file__).resolve().parents[1]
O = R / 'runs' / 'memory-repair-415'

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--dataset-manifest', type=Path, required=True)
    p.add_argument('--split', default='dev')
    p.add_argument('--indices', default='0')
    a = p.parse_args()
    if a.output.exists():
        raise SystemExit('output directory already exists')
    if not a.dataset_manifest.is_file():
        raise SystemExit('dataset manifest not found')
    subprocess.run([
        str(R / '.envs' / 'trackon312' / 'bin' / 'python'), '-u',
        str(O / 'evaluate_resource.py'),
        '--checkpoint', str(O / 'train_seed0' / 'best.pt'),
        '--output', str(a.output.resolve()),
        '--manifest', str(a.dataset_manifest.resolve()),
        '--split', a.split,
        '--indices', a.indices,
    ], cwd=R, check=True)

if __name__ == '__main__':
    main()
