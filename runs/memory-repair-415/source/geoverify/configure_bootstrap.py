"""Generate a verified, training-only configuration for the temporary DINOv2 path.

No inference configuration or placeholder calibrator is created. Source commits
are lock declarations, not source-tree audits; GPU readiness is checked separately.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path

from .cache import validate_provenance
from .datasets import file_sha256
from .interpreter import interpreter_path
from .runtime import tracker_configuration_sha256


def configure(root, output, geometry_physical_gpu, device='cuda:0',metric_only=False):
    root = Path(root).resolve(strict=True)
    output = Path(output).absolute()
    if output.exists():
        raise FileExistsError(output)
    if not str(geometry_physical_gpu).isdigit():
        raise ValueError('Explicit nonnegative physical GPU index required')
    if not device.startswith('cuda:') or not device[5:].isdigit():
        raise ValueError('Explicit CUDA device required')
    lock_path = root / 'configs/assets.lock.json'
    lock = json.loads(lock_path.read_text())
    keys = ('trackon2_dinov2', 'dinov2_small', 'dinov2_config', 'verifier',
            'track4world_da3', 'da3_nested', 'da3_config')
    if metric_only:keys=tuple(k for k in keys if k!='da3_nested')
    paths, hashes = {}, {}
    for key in keys:
        entry = lock['downloads'][key]
        path = (root / entry['path']).resolve(strict=True)
        if root not in path.parents:
            raise ValueError('Asset outside workspace: ' + key)
        if path.stat().st_size != entry['bytes']:
            raise ValueError('Incomplete asset: ' + key)
        digest = file_sha256(path)
        if 'sha256' in entry:
            if digest != entry['sha256']:
                raise ValueError('Asset digest mismatch: ' + key)
        elif 'git_blob_sha1' in entry:
            content = path.read_bytes()
            blob = b'blob ' + str(len(content)).encode() + b'\0' + content
            if hashlib.sha1(blob).hexdigest() != entry['git_blob_sha1']:
                raise ValueError('Asset Git blob mismatch: ' + key)
        else:
            raise ValueError('Missing trusted digest: ' + key)
        paths[key], hashes[key] = path, digest
    metric_provenance=None
    if metric_only:
        from .metric_bundle import audit_bundle
        selection=json.loads((root/'configs/da3.metric.lock.json').read_text())
        if selection['source_sha256']!=lock['downloads']['da3_nested']['sha256']:
            raise ValueError('Metric selection differs from official source lock')
        directory=root/'weights/da3nested-metric'
        metric_provenance=audit_bundle(directory,selection)
        if file_sha256(directory/'config.json')!=hashes['da3_config']:
            raise ValueError('Metric bundle config differs from official DA3 config')
        paths['da3_nested']=directory/'model.safetensors'
        hashes['da3_nested']=file_sha256(paths['da3_nested'])
    for filename in ('third_party/track_on', 'third_party/track4world'):
        if not (root / filename).is_dir():
            raise FileNotFoundError(root / filename)
    python = interpreter_path(root / '.envs/track4world311/bin/python')
    if not python.is_file() or not os.access(python, os.X_OK):
        raise ValueError('Geometry interpreter missing or not executable')
    sources = lock['sources']
    provenance = {
        'track_on': dict(commit=sources['track_on']['commit'],
                         weights_sha256=hashes['trackon2_dinov2'],
                         variant='trackon2_dinov2', backbone='dinov2_s',
                         backbone_weights_sha256=hashes['dinov2_small'],
                         config_sha256=tracker_configuration_sha256(
                             'trackon2_dinov2', paths['dinov2_config'])),
        'verifier': dict(commit=sources['track_on']['commit'],
                         weights_sha256=hashes['verifier']),
        'track4world': dict(commit=sources['track4world']['commit'],
                            weights_sha256=hashes['track4world_da3'],
                            backbone_weights_sha256=hashes['da3_nested'],
                            backbone_config_sha256=hashes['da3_config']),
    }
    validate_provenance(provenance)
    if metric_provenance is not None:
        provenance['track4world'].update(metric_subset_manifest_sha256=file_sha256(paths['da3_nested'].parent/'metric_source.json'),
                                         original_da3_source_sha256=metric_provenance['selection']['source_sha256'])
    config = dict(schema_version=1, purpose='cache_bootstrap',
                  baseline_provenance=provenance, device=device,
                  geometry_physical_gpu=str(geometry_physical_gpu),
                  asset_lock_sha256=file_sha256(lock_path),
                  trackon_root=str(root / 'third_party/track_on'),
                  trackon_checkpoint=str(paths['trackon2_dinov2']),
                  backbone_directory=str(paths['dinov2_small'].parent),
                  verifier_checkpoint=str(paths['verifier']),
                  geometry_python=str(python),
                  geometry_root=str(root / 'third_party/track4world'),
                  geometry_checkpoint=str(paths['track4world_da3']),
                  geometry_backbone_directory=str(paths['da3_nested'].parent),
                  ipc_root=str(root / 'runs/ipc'),
                  pipeline=dict(geometry_window=16, geometry_stride=4,
                                verifier_window=24, tau_seconds=.25,
                                max_geometry_age=1.))
    config['geometry_intrinsics_protocol']=dict(
        version='da3_per_frame_resize_half_pixel_unpad_v1',
        implementation_sha256=file_sha256(Path(__file__).with_name('geometry_intrinsics.py')),
        adapter_sha256=file_sha256(Path(__file__).with_name('track4world_adapter.py')))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as stream:
        json.dump(config, stream, indent=2)
        stream.write('\n')
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', default='.')
    parser.add_argument('--output', required=True)
    parser.add_argument('--geometry-physical-gpu', required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--metric-only',action='store_true',help='Use exact metric tensors plus complete official Track4World weights')
    args = parser.parse_args()
    configure(args.root, args.output, args.geometry_physical_gpu, args.device,args.metric_only)


if __name__ == '__main__':
    main()
