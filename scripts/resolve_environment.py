"""Resolve the complete pinned upstream environment without installing packages.

The observed proxy handshake can exceed uv's default 10-second connect timeout.
Both timeouts are set explicitly; HTTPS verification and package hashes remain on.
Reference: https://docs.astral.sh/uv/configuration/environment/
"""
import argparse
import os
from pathlib import Path
import subprocess


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--component',choices=('trackon','track4world'),required=True)
    parser.add_argument('--timeout',type=int,default=120)
    parser.add_argument('--offline',action='store_true',help='Diagnose missing cached metadata without network')
    parser.add_argument('--allow-source-builds',action='store_true',
                        help='Allow isolated build metadata for upstream packages without published wheels')
    args=parser.parse_args()
    if args.timeout<1:
        parser.error('--timeout must be positive')
    root=Path(__file__).resolve().parent.parent
    uv=root/'.envs/trackon/bin/uv'
    python=root/('.envs/trackon312/bin/python' if args.component=='trackon' else '.envs/track4world311/bin/python')
    if not uv.is_file() or not python.is_file():
        parser.error('Bootstrap uv and the isolated target Python must exist first')
    command=[str(uv),'pip','compile','environments/'+args.component+'-full.in',
             '--python',str(python),'--format','pylock.toml',
             '--output-file','environments/pylock.'+args.component+'-full.toml',
             '--index','https://download.pytorch.org/whl/cu121',
             '--index-strategy','unsafe-best-match','--find-links','.wheelhouse']
    if args.offline:
        command.append('--offline')
    if not args.allow_source_builds:
        command.append('--no-build')
    environment=os.environ.copy()
    environment.update(UV_HTTP_CONNECT_TIMEOUT=str(args.timeout),UV_HTTP_TIMEOUT=str(args.timeout))
    # No retry loop: a terminal error must be inspected rather than hidden.
    raise SystemExit(subprocess.run(command,cwd=root,env=environment).returncode)


if __name__=='__main__':
    main()
