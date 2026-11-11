"""Bind training and inference to the same raw-conditioned evidence implementation."""
from pathlib import Path
import hashlib

def protocol():
    return dict(version='raw_trackon_evidence_history_v1',history='raw_trackon',
                pipeline_sha256=hashlib.sha256(Path(__file__).with_name('pipeline.py').read_bytes()).hexdigest())

def validate(value):
    if value != protocol():
        raise ValueError('Input history protocol mismatch; rebuild cache/config and retrain')
    return dict(value)

def validate_runtime(config):
    value=validate(config.get('input_history_protocol'))
    if config.get('pipeline',{}).get('evidence_history_policy') != value['history']:
        raise ValueError('Runtime evidence history does not match trained input protocol')
    return value
