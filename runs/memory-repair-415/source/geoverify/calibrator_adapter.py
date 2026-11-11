"""Load only a trained calibrator with exactly matching baseline provenance."""
from pathlib import Path
import hashlib
import os, uuid
import numpy as np
from geoverify.probability_boundary import bounded_probability
import torch
from geoverify.calibration import GeometryCalibrationHead,FEATURE_NAMES,HEAD_ARCHITECTURE
from geoverify.cache import validate_provenance
from geoverify.appearance_cache import validate_descriptor_protocol


class CalibratorAdapter:
    def __init__(self,checkpoint,baseline_provenance,device='cuda:0',geometry_protocol=None,evidence_age_protocol=None):
        validate_provenance(baseline_provenance)
        checkpoint=Path(checkpoint).resolve()
        state=torch.load(str(checkpoint),map_location='cpu',weights_only=True)
        if state.get('head_architecture')!=HEAD_ARCHITECTURE:
            raise ValueError('Calibrator head architecture mismatch; retrain with visibility residual head')
        if state.get('schema_version')!=3 or state.get('feature_names')!=list(FEATURE_NAMES):
            raise ValueError('Calibrator feature schema mismatch')
        if not isinstance(state.get('epoch'),int) or state['epoch']<1:
            raise ValueError('Formal inference requires a completed training epoch')
        if state.get('baseline_provenance')!=baseline_provenance:
            raise ValueError('Calibrator baseline provenance mismatch; rebuild cache and retrain')
        if not state.get('manifest_sha256'):
            raise ValueError('Missing training cache provenance')
        if geometry_protocol is not None and state.get('geometry_intrinsics_protocol')!=geometry_protocol:
            raise ValueError('Calibrator geometry protocol mismatch; rebuild cache and retrain')
        if evidence_age_protocol is not None and state.get('evidence_age_protocol')!=evidence_age_protocol:
            raise ValueError('Calibrator evidence age protocol mismatch; rebuild cache and retrain')
        from .input_history_protocol import validate as validate_input_history
        self.input_history_protocol=validate_input_history(state.get('input_history_protocol'))
        self.descriptor_protocol=validate_descriptor_protocol(state.get('descriptor_protocol'))
        if self.descriptor_protocol['implementation_sha256']!=hashlib.sha256(Path(__file__).with_name('appearance_descriptor.py').read_bytes()).hexdigest():
            raise ValueError('Descriptor implementation mismatch')
        if self.descriptor_protocol['provider_weights_sha256']!=baseline_provenance['verifier']['weights_sha256']:
            raise ValueError('Descriptor provider mismatch')
        self.appearance_mode=state.get('appearance_mode')
        self.model=GeometryCalibrationHead(appearance_mode=self.appearance_mode).to(device).eval()
        self.requires_descriptor=True
        self.model.load_state_dict(state['model'],strict=True)
        if not all(bool(torch.isfinite(p).all()) for p in self.model.state_dict().values()):
            raise ValueError('Non-finite calibrator checkpoint')
        if not bool((self.model.feature_std>0).all() and (self.model.appearance_std>0).all()):
            raise ValueError('Invalid training feature normalization')
        self.model.requires_grad_(False)
        self.device=device
        self.provenance=baseline_provenance
        self.training_source_sha256=state.get('training_source_sha256',[])

    def predict(self,features,visual_logits,packet):
        def tensor(value):
            return torch.as_tensor(value,device=self.device)
        if 'appearance_descriptor' not in packet:
            raise ValueError('Missing current-frame visual descriptor')
        with torch.inference_mode():
            output=self.model(tensor(features).float(),tensor(visual_logits).float(),
                              tensor(packet['candidates']).float(),
                              tensor(packet['candidate_visibility']).float(),
                              tensor(packet['valid']).bool(),tensor(packet['appearance_descriptor']).float())
        if getattr(self,'benefit_gate',None) is not None:
            with torch.inference_mode():
                output=self.benefit_gate.apply(output,tensor(features).float(),tensor(visual_logits).float(),
                                               tensor(packet['candidates']).float(),tensor(packet['valid']).bool())
        result = {k:v.cpu().numpy() for k,v in output.items()}
        for key in ('visibility','confidence'):
            values=result[key]
            if not np.isfinite(values).all() or ((values<0)|(values>1)).any():
                directory=os.environ.get('GEOVERIFY_PROBABILITY_AUDIT_DIR')
                if directory:
                    path=Path(directory);path.mkdir(parents=True,exist_ok=True)
                    np.savez_compressed(path/(uuid.uuid4().hex+'.npz'),
                        features=features,visual_logits=visual_logits,
                        candidates=packet['candidates'],candidate_visibility=packet['candidate_visibility'],
                        valid=packet['valid'],appearance_descriptor=packet['appearance_descriptor'],**{'output_'+k:v for k,v in result.items()})
                result[key]=bounded_probability(values,key)
                print('Probability boundary correction: '+key+' min='+str(values.min())+' max='+str(values.max()),flush=True)
        return result
