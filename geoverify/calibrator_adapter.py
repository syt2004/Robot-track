"""Load only a trained calibrator with exactly matching baseline provenance."""
from pathlib import Path
import torch
from geoverify.calibration import GeometryCalibrationHead,FEATURE_NAMES
from geoverify.cache import validate_provenance


class CalibratorAdapter:
    def __init__(self,checkpoint,baseline_provenance,device='cuda:0',geometry_protocol=None):
        validate_provenance(baseline_provenance)
        checkpoint=Path(checkpoint).resolve()
        state=torch.load(str(checkpoint),map_location='cpu',weights_only=True)
        if state.get('schema_version')!=1 or state.get('feature_names')!=list(FEATURE_NAMES):
            raise ValueError('Calibrator feature schema mismatch')
        if not isinstance(state.get('epoch'),int) or state['epoch']<1:
            raise ValueError('Formal inference requires a completed training epoch')
        if state.get('baseline_provenance')!=baseline_provenance:
            raise ValueError('Calibrator baseline provenance mismatch; rebuild cache and retrain')
        if not state.get('manifest_sha256'):
            raise ValueError('Missing training cache provenance')
        if geometry_protocol is not None and state.get('geometry_intrinsics_protocol')!=geometry_protocol:
            raise ValueError('Calibrator geometry protocol mismatch; rebuild cache and retrain')
        self.model=GeometryCalibrationHead().to(device).eval()
        self.model.load_state_dict(state['model'],strict=True)
        if not all(bool(torch.isfinite(p).all()) for p in self.model.state_dict().values()):
            raise ValueError('Non-finite calibrator checkpoint')
        if not bool((self.model.feature_std>0).all()):
            raise ValueError('Invalid training feature normalization')
        self.model.requires_grad_(False)
        self.device=device
        self.provenance=baseline_provenance
        self.training_source_sha256=state.get('training_source_sha256',[])

    def predict(self,features,visual_logits,packet):
        def tensor(value):
            return torch.as_tensor(value,device=self.device)
        with torch.inference_mode():
            output=self.model(tensor(features).float(),tensor(visual_logits).float(),
                              tensor(packet['candidates']).float(),
                              tensor(packet['candidate_visibility']).float(),
                              tensor(packet['valid']).bool())
        return {k:v.cpu().numpy() for k,v in output.items()}
