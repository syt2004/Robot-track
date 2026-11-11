"""Assemble actual baseline adapters; no fixture or fallback model paths."""
import hashlib
import json
from pathlib import Path
from .cache import validate_provenance
from .datasets import file_sha256
from .profiles import tracker_args
from .interpreter import interpreter_path


def tracker_configuration_sha256(variant,backbone_config):
    value=dict(tracker_args=vars(tracker_args(variant)),
               backbone_config_sha256=file_sha256(backbone_config),support_grid_size=0)
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


class RuntimeFactory:
    """Reuse frozen parent models, create an owned geometry worker per session.

    Paths in a schema-1 JSON config resolve relative to that config. Provenance
    commits describe the supplied source checkouts; source-tree auditing remains
    a separate prerequisite. This class verifies actual weight/config hashes.
    Track4World's nested DA3 backbone must be a checksum-verified local folder;
    its path is forwarded to the isolated worker, with no Hub download on load.
    """
    def __init__(self,config_path,purpose='inference'):
        config_path=Path(config_path).resolve()
        self.config=json.loads(config_path.read_text())
        c=self.config
        if c.get('schema_version')!=1:
            raise ValueError('Expected runtime config schema 1')
        if purpose not in ('inference','cache_bootstrap') or c.get('purpose','inference')!=purpose:
            raise ValueError('Runtime purpose mismatch; bootstrap is training-only')
        protocol=c.get('geometry_intrinsics_protocol',{})
        expected=dict(version='da3_per_frame_resize_half_pixel_unpad_v1',
                      implementation_sha256=file_sha256(Path(__file__).with_name('geometry_intrinsics.py')),
                      adapter_sha256=file_sha256(Path(__file__).with_name('track4world_adapter.py')))
        if protocol!=expected:
            raise ValueError('Geometry intrinsics protocol mismatch; rebuild cache/config and retrain')
        self.purpose=purpose
        self.provenance=c['baseline_provenance']
        validate_provenance(self.provenance)
        required=('trackon_root','trackon_checkpoint','backbone_directory','verifier_checkpoint',
                  'geometry_python','geometry_root','geometry_checkpoint','geometry_backbone_directory','ipc_root')
        if purpose=='inference':
            required+=('calibrator_checkpoint',)
        self.paths={key:(interpreter_path(config_path.parent/c[key]) if key=='geometry_python'
                         else (config_path.parent/c[key]).resolve()) for key in required}
        p=self.paths
        for key,name in (('trackon_checkpoint','track_on'),('verifier_checkpoint','verifier'),
                         ('geometry_checkpoint','track4world')):
            if file_sha256(p[key])!=self.provenance[name]['weights_sha256']:
                raise ValueError('Weight identity mismatch: '+key)
        tracker=self.provenance['track_on']
        if file_sha256(p['backbone_directory']/'model.safetensors')!=tracker['backbone_weights_sha256']:
            raise ValueError('Frozen backbone weight mismatch')
        if tracker_configuration_sha256(tracker['variant'],p['backbone_directory']/'config.json')!=tracker['config_sha256']:
            raise ValueError('Tracker configuration mismatch')
        geometry=self.provenance['track4world']
        for filename,key in (('model.safetensors','backbone_weights_sha256'),('config.json','backbone_config_sha256')):
            if file_sha256(p['geometry_backbone_directory']/filename)!=geometry.get(key):
                raise ValueError('Geometry backbone identity mismatch: '+filename)
        metric_manifest=p['geometry_backbone_directory']/'metric_source.json'
        if metric_manifest.exists() or geometry.get('metric_subset_manifest_sha256'):
            if not metric_manifest.is_file() or file_sha256(metric_manifest)!=geometry.get('metric_subset_manifest_sha256'):
                raise ValueError('Metric subset provenance mismatch')
            from .metric_bundle import audit_bundle
            selection=json.loads(metric_manifest.read_text())['selection']
            if selection['source_sha256']!=geometry.get('original_da3_source_sha256'):
                raise ValueError('Metric original source identity mismatch')
            audit_bundle(p['geometry_backbone_directory'],selection)
        if purpose=='inference' and file_sha256(p['calibrator_checkpoint'])!=c['calibrator_sha256']:
            raise ValueError('Calibrator weight mismatch')
        # Import GPU libraries only after input identities pass validation.
        from .adapters import TrackOnAdapter,VerifierAdapter
        if purpose=='inference':
            from .calibrator_adapter import CalibratorAdapter
            self.calibrator=CalibratorAdapter(p['calibrator_checkpoint'],self.provenance,c['device'],geometry_protocol=protocol)
        else:
            from .bootstrap import VerifierBootstrapPolicy
            self.calibrator=VerifierBootstrapPolicy()
        self.trackon=TrackOnAdapter(p['trackon_root'],p['trackon_checkpoint'],c['device'],
                                   tracker['variant'],p['backbone_directory'])
        self.verifier=VerifierAdapter(p['trackon_root'],p['verifier_checkpoint'],c['device'])
        self.active=None

    def __call__(self,clock):
        from .async_geometry import AsyncGeometryWorker
        from .pipeline import CausalGeoVerifyTracker
        if self.active is not None and not self.active.closed:
            raise RuntimeError('Shared Track-On state permits only one active session')
        c,p=self.config,self.paths
        worker=AsyncGeometryWorker(p['geometry_python'],p['geometry_root'],p['geometry_checkpoint'],
                                   p['ipc_root'],c['geometry_physical_gpu'],p['geometry_backbone_directory'],clock=clock)
        self.active=worker
        try:
            worker.wait_ready(timeout=c.get('geometry_init_timeout',600.),
                              on_wait=lambda seconds,log:print('Geometry loading %.0fs; log %s'%(seconds,log),flush=True))
            return CausalGeoVerifyTracker(self.trackon,self.verifier,self.calibrator,worker,
                                         clock=clock,**c.get('pipeline',{}))
        except BaseException:
            worker.close()
            raise
