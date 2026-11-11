"""Run a module from a hash-verified source snapshot, preserving caller cwd.

Example: python scripts/run_frozen_module.py --source RUN/source \
    --manifest RUN/source_manifest.json geoverify.promote_runtime --help
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import runpy
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True, type=Path)
    parser.add_argument('--manifest', required=True, type=Path)
    parser.add_argument('module')
    parser.add_argument('args', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    source = args.source.resolve(strict=True)
    manifest = json.loads(args.manifest.read_text())
    if not isinstance(manifest, dict) or not manifest:
        parser.error('Expected a nonempty relative-path to SHA256 manifest')
    for relative, expected in manifest.items():
        path = (source / relative).resolve(strict=True)
        if not path.is_relative_to(source):
            parser.error('Manifest path escapes source: ' + relative)
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            parser.error('Frozen source SHA256 mismatch: ' + relative)
    if not re.fullmatch(r'[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*', args.module):
        parser.error('Expected a dotted module name')
    relative = args.module.replace('.', '/') + '.py'
    if relative not in manifest:
        parser.error('Executable module is not bound by manifest: ' + relative)
    # PYTHONPATH alone is insufficient: -m gives the caller cwd precedence.
    sys.path.insert(0, str(source))
    spec = importlib.util.find_spec(args.module)
    if spec is None or Path(spec.origin).resolve() != (source / relative).resolve():
        parser.error('Module resolved outside the verified snapshot')
    sys.argv = [str(source / relative), *args.args]
    runpy.run_module(args.module, run_name='__main__', alter_sys=True)


if __name__ == '__main__':
    main()
