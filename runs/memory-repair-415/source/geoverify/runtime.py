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
        from .evidence_clock import protocol as clock_protocol
        age_protocol=clock_protocol(c.get('pipeline',{}).get('evidence_age_clock','availability'))
        if c.get('evidence_age_protocol')!=age_protocol:raise ValueError('Evidence age protocol mismatch; rebuild cache and retrain')
        protocol=c.get('geometry_intrinsics_protocol',{})
        expected=dict(version='da3_per_frame_resize_half_pixel_unpad_v1',
                      implementation_sha256=file_sha256(Path(__file__).with_name('geometry_intrinsics.py')),
                      adapter_sha256=file_sha256(Path(__file__).with_name('track4world_adapter.py')))
        if protocol!=expected:
            raise ValueError('Geometry intrinsics protocol mismatch; rebuild cache/config and retrain')
        from .input_history_protocol import validate_runtime
        input_history_protocol=validate_runtime(c)
        self.purpose=purpose
        visibility_policy=c.get('visibility_policy')
        if visibility_policy not in (None,'independent_raw_history_v1'):
            raise ValueError('Unknown visibility policy')
        if visibility_policy is not None and purpose!='inference':
            raise ValueError('Independent visibility composition is inference-only')
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
        visibility_stream=None
        if visibility_policy is not None:
            from .visibility_stream import RawVisibilityStream
            visibility_path=(config_path.parent/c['visibility_checkpoint']).resolve()
            visibility_stream=RawVisibilityStream(visibility_path,c['visibility_sha256'],self.provenance)
        # Import GPU libraries only after input identities pass validation.
        from .adapters import TrackOnAdapter,VerifierAdapter
        if purpose=='inference':
            from .calibrator_adapter import CalibratorAdapter
            self.calibrator=CalibratorAdapter(p['calibrator_checkpoint'],self.provenance,c['device'],geometry_protocol=protocol,evidence_age_protocol=age_protocol)
            if c.get('geometry_benefit_gate'):
                import torch
                from .benefit_gate import BenefitGate
                gate_config=c['geometry_benefit_gate']
                gate_path=(config_path.parent/gate_config['checkpoint']).resolve()
                if file_sha256(gate_path)!=gate_config['sha256']:raise ValueError('Benefit gate checksum mismatch')
                state=torch.load(gate_path,map_location='cpu',weights_only=True)
                if state['base_sha256']!=c['calibrator_sha256']:raise ValueError('Benefit gate baseline mismatch')
                if state['architecture']!='counterfactual_geometry_benefit_gate_v1':raise ValueError('Benefit gate architecture mismatch')
                if c['pipeline']['evidence_history_policy']!='raw_trackon':raise ValueError('Benefit gate requires raw evidence history')
                gate=BenefitGate().to(c['device']).eval();gate.load_state_dict(state['model']);gate.requires_grad_(False)
                self.calibrator.benefit_gate=gate
                self.calibrator.benefit_training_source_sha256=state['training_source_sha256']

        else:
            from .bootstrap import VerifierBootstrapPolicy
            self.calibrator=VerifierBootstrapPolicy()
        if purpose=='inference' and self.calibrator.input_history_protocol!=input_history_protocol:
            raise ValueError('Runtime and checkpoint input history differ')
        if visibility_stream is not None:
            if set(visibility_stream.training_source_sha256)!=set(self.calibrator.training_source_sha256):
                raise ValueError('Visibility/spatial training source identities differ')
            self.calibrator.visibility_stream=visibility_stream
            self.calibrator.output_kind='geoverify_independent_raw_visibility_v10'
        if getattr(self.calibrator,'benefit_training_source_sha256',None):
            self.calibrator.training_source_sha256=sorted(set(self.calibrator.training_source_sha256)|set(self.calibrator.benefit_training_source_sha256))
        self.trackon=TrackOnAdapter(p['trackon_root'],p['trackon_checkpoint'],c['device'],
                                   tracker['variant'],p['backbone_directory'])
        if c.get('memory_repair'):
            import torch
            from .memory_repair import GeometryMemoryRepair
            from .memory_adapter import RepairingTrackOn
            cfg=c['memory_repair'];path=Path(cfg['checkpoint'])
            if file_sha256(path)!=cfg['sha256']:raise ValueError('Memory repair checksum mismatch')
            state=torch.load(path,map_location='cpu',weights_only=True)
            if state.get('architecture')!='verified_geometry_shadow_memory_v4' or cfg.get('architecture')!=state['architecture']:raise ValueError('Bounded memory architecture mismatch')
            if state.get('maximum_relative_change')!=.25:raise ValueError('Bounded memory radius mismatch')
            repair=GeometryMemoryRepair(state['dimension'],maximum_relative_change=state['maximum_relative_change']).to(c['device']).eval()
            if file_sha256(Path(cfg['verified_manifest']))!=state['verified_manifest_sha256']:raise ValueError('Historical verification training cache identity mismatch')
            repair.load_state_dict(state['model']);repair.requires_grad_(False)
            self.trackon=RepairingTrackOn(self.trackon,repair,audit_path=cfg.get('audit_path'))
            if state.get('train_sources'):
                if file_sha256(Path(cfg['sequence_plan']))!=state['plan_sha256']:raise ValueError('Memory training source plan mismatch')
                sequence_plan=json.loads(Path(cfg['sequence_plan']).read_text())
                repair_sources=[]
                for name in state['train_sources']:
                    dataset=json.loads(Path(sequence_plan['manifest_by_source'][name]).read_text())
                    repair_sources.extend(scene['annotation']['sha256'] for scene in dataset['scenes'] if scene['name']==name)
            else:
                repair_sources=[source['sha256'] for source in state.get('sources',[])]
            self.calibrator.training_source_sha256=sorted(set(self.calibrator.training_source_sha256)|set(repair_sources))
        if purpose=='inference':
            from .appearance_descriptor import DescriptorVerifier
            if c.get('appearance_descriptor_protocol')!=self.calibrator.descriptor_protocol or c.get('appearance_mode')!=self.calibrator.appearance_mode:
                raise ValueError('Runtime visual descriptor or ablation mode mismatch')
            self.verifier=DescriptorVerifier(p['trackon_root'],p['verifier_checkpoint'],c['device'])
        else:
            self.verifier=VerifierAdapter(p['trackon_root'],p['verifier_checkpoint'],c['device'])
        if c.get('memory_repair'):self.trackon.verifier=self.verifier
        self.active=None

    def __call__(self,clock):
        from .async_geometry import AsyncGeometryWorker
        from .pipeline import CausalGeoVerifyTracker
        if self.active is not None and not self.active.closed:
            raise RuntimeError('Shared Track-On state permits only one active session')
        c,p=self.config,self.paths
        worker=AsyncGeometryWorker(p['geometry_python'],p['geometry_root'],p['geometry_checkpoint'],
                                   p['ipc_root'],c['geometry_physical_gpu'],p['geometry_backbone_directory'],clock=clock,
                                   minimum_lag_frames=c.get('geometry_minimum_lag_frames',0))
        self.active=worker
        try:
            worker.wait_ready(timeout=c.get('geometry_init_timeout',600.),
                              on_wait=lambda seconds,log:print('Geometry loading %.0fs; log %s'%(seconds,log),flush=True))
            if c.get('memory_repair'):
                from .memory_adapter import MemoryGeometryWorker
                worker=MemoryGeometryWorker(worker,self.trackon)
                from .shadow_pipeline import ShadowEvidenceStream,ShadowHistoryPipeline
                if c['memory_repair'].get('history_reference')!='independent_shadow_history_v1':raise ValueError('Explicit shadow history protocol required')
                if file_sha256(Path(__file__).with_name('shadow_pipeline.py'))!=c['memory_repair'].get('history_wrapper_sha256'):raise ValueError('Shadow history wrapper checksum mismatch')
                return ShadowHistoryPipeline(ShadowEvidenceStream(self.trackon),self.verifier,self.calibrator,worker,clock=clock,**c.get('pipeline',{}))
            return CausalGeoVerifyTracker(self.trackon,self.verifier,self.calibrator,worker,
                                         clock=clock,**c.get('pipeline',{}))
        except BaseException:
            worker.close()
            raise
