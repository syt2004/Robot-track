"""Bind a trained calibrator to the exact bootstrap baseline configuration.

This creates an inference configuration, not a claim that GPU assets are ready.
RuntimeFactory still verifies every baseline weight when the configuration runs.
"""
import argparse
import json
from pathlib import Path
from .calibrator_adapter import CalibratorAdapter
from .datasets import file_sha256
from .interpreter import interpreter_path


def promote(bootstrap_config,checkpoint,output):
    source=Path(bootstrap_config).resolve(strict=True)
    checkpoint=Path(checkpoint).resolve(strict=True)
    output=Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    config=json.loads(source.read_text())
    if config.get('schema_version')!=1 or config.get('purpose')!='cache_bootstrap':
        raise ValueError('Expected explicit schema-1 cache_bootstrap config')
    # Exercise the same strict schema/provenance/model loader as formal inference.
    CalibratorAdapter(checkpoint,config['baseline_provenance'],device='cpu',
                      geometry_protocol=config.get('geometry_intrinsics_protocol'))
    paths=('trackon_root','trackon_checkpoint','backbone_directory','verifier_checkpoint',
           'geometry_python','geometry_root','geometry_checkpoint','geometry_backbone_directory','ipc_root')
    for name in paths:
        path=source.parent/config[name]
        config[name]=str(interpreter_path(path) if name=='geometry_python' else path.resolve())
    config.update(purpose='inference',calibrator_checkpoint=str(checkpoint),
                  calibrator_sha256=file_sha256(checkpoint),
                  promoted_from_config_sha256=file_sha256(source))
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x') as stream:
        json.dump(config,stream,indent=2)
        stream.write('\n')
    return output


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bootstrap-config',required=True)
    parser.add_argument('--checkpoint',required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    promote(args.bootstrap_config,args.checkpoint,args.output)


if __name__=='__main__':
    main()
