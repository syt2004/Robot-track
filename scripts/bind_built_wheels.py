"""Bind verified source-built wheels without re-resolving any package versions."""
import argparse
from email.parser import Parser
import json
from pathlib import Path
import re
import tomllib
import zipfile
from download_verified import file_digest


def bind(source_lock,build_manifest,wheelhouse,output):
    source_lock,build_manifest,wheelhouse,output=map(Path,(source_lock,build_manifest,wheelhouse,output))
    if output.exists():
        raise FileExistsError(output)
    original=source_lock.read_text()
    lock=tomllib.loads(original)
    builds=json.loads(build_manifest.read_text())
    if builds.get('state')!='complete' or builds.get('source_lock_sha256')!=file_digest(source_lock):
        raise ValueError('Completed build manifest must match exact source lock')
    by_name={(row['name'],row['version']):row for row in builds['builds']}
    if len(by_name)!=len(builds['builds']):
        raise ValueError('Duplicate built package')
    sections=original.split('\n[[packages]]\n')
    if len(sections)!=len(lock['packages'])+1:
        raise ValueError('Unexpected source lock layout')
    normalize=lambda value:re.sub(r'[-_.]+','-',value).lower()
    for index,package in enumerate(lock['packages'],1):
        if package.get('wheels'):
            continue
        row=by_name.pop((package['name'],package['version']))
        if row['source_sha256']!=package['sdist']['hashes']['sha256'] or row['source_url']!=package['sdist']['url']:
            raise ValueError('Built wheel source differs from locked archive')
        if Path(row['wheel']).name!=row['wheel'] or not row['wheel'].endswith('.whl'):
            raise ValueError('Unsafe built wheel path')
        wheel=wheelhouse/row['wheel']
        if file_digest(wheel)!=row['wheel_sha256']:
            raise ValueError('Built wheel hash mismatch')
        with zipfile.ZipFile(wheel) as archive:
            metadata=[name for name in archive.namelist() if name.endswith('.dist-info/METADATA')]
            if len(metadata)!=1 or archive.testzip() is not None:
                raise ValueError('Invalid built wheel archive')
            info=Parser().parsestr(archive.read(metadata[0]).decode())
        if normalize(info['Name'])!=normalize(package['name']) or info['Version']!=package['version']:
            raise ValueError('Built wheel package identity mismatch')
        sections[index]+='\nwheels = [{ url = '+json.dumps(wheel.resolve().as_uri())+', hashes = { sha256 = '+json.dumps(row['wheel_sha256'])+' } }]\n'
    if by_name:
        raise ValueError('Build manifest contains packages not required by source lock')
    content='\n[[packages]]\n'.join(sections)
    checked=tomllib.loads(content)
    if any(not package.get('wheels') for package in checked['packages']):
        raise ValueError('A locked package still has no wheel')
    # Preserve all original source records and all package versions verbatim.
    content='# Source build manifest SHA256: '+file_digest(build_manifest)+'\n'+content
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x') as stream:
        stream.write(content)
    return output


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-lock',required=True)
    parser.add_argument('--build-manifest',required=True)
    parser.add_argument('--wheelhouse',default='.wheelhouse')
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    print(bind(args.source_lock,args.build_manifest,args.wheelhouse,args.output))


if __name__=='__main__':
    main()
