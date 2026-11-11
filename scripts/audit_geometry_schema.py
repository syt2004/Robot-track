"""Inspect real architecture and checkpoint metadata without loading weight values.

Run in the Track4World environment. This is not checksum or inference acceptance.
Only a restricted tensor-metadata unpickler is used on the checkpoint prefix.
"""
import collections
import json
from pathlib import Path
import pickle
import struct
import sys
from types import SimpleNamespace
import torch
root=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(root));sys.path.insert(0,str(root/'third_party/track4world'))
from geoverify.local_backbone import checkpoint_convnext_initializer
from track4world.nets import model as module


class MetadataUnpickler(pickle.Unpickler):
    def persistent_load(self,pid):return None
    def find_class(self,name,attribute):
        if (name,attribute)==('collections','OrderedDict'):return collections.OrderedDict
        if name=='torch' and attribute in ('FloatStorage','HalfStorage','LongStorage','IntStorage','ByteStorage','BoolStorage','BFloat16Storage'):
            return attribute
        if (name,attribute)==('torch._utils','_rebuild_tensor_v2'):
            return lambda storage,offset,size,stride,*rest:list(size)
        raise ValueError('Unsupported checkpoint metadata global')


def main():
    original_register=torch.nn.Module.register_parameter
    original_class=module.DepthAnything3
    def register_meta(self,name,param):
        original_register(self,name,param)
        if param is not None:
            self._parameters[name]=torch.nn.Parameter(param.to('meta'),requires_grad=param.requires_grad)
    torch.nn.Module.register_parameter=register_meta
    try:
        backbone=original_class(model_name='da3nested-giant-large')
        module.DepthAnything3=SimpleNamespace(from_pretrained=lambda *a,**kw:backbone)
        with checkpoint_convnext_initializer(module):
            model=module.Track4World(**json.loads((root/'third_party/track4world/track4world/config/eval/v1.json').read_text())['model'],
                                      seqlen=16,use_3d=True,use_model='depthanythingv3')
    finally:
        module.DepthAnything3=original_class;torch.nn.Module.register_parameter=original_register
    path=root/'weights/track4world/track4world_da3.pth'
    if not path.exists():path=path.with_name(path.name+'.part')
    with path.open('rb') as stream:
        header=struct.unpack('<4s5H3I2H',stream.read(30))
        name=stream.read(header[-2]);stream.read(header[-1])
        if header[0]!=b'PK\x03\x04' or not name.endswith(b'/data.pkl') or header[3]!=0:
            raise ValueError('Unsupported checkpoint ZIP metadata layout')
        state=MetadataUnpickler(stream).load()
    remapped={k.replace('backbone.model.','backbone.model.da3.',1)
        if k.startswith('backbone.model.') and not k.startswith(('backbone.model.da3.','backbone.model.da3_metric.'))
        else k:v for k,v in state.items()}
    expected={k:list(v.shape) for k,v in model.state_dict().items()}
    metric={k for k in expected if k.startswith('backbone.model.da3_metric.')}
    missing=set(expected)-set(remapped)
    report=dict(expected_model_keys=len(expected),checkpoint_keys=len(remapped),expected_metric_missing=len(metric),
        context_encoder_keys=sum(k.startswith('ctx_encoder.') for k in expected),
        missing_nonmetric=sorted(missing-metric),metric_keys_present_in_tracking_checkpoint=sorted(metric-missing),
        unexpected=sorted(set(remapped)-set(expected)),
        shape_mismatch=[k for k in expected if k in remapped and expected[k]!=remapped[k]],
        weights_loaded=False,full_checkpoint_digest_verified=False,meta_structure_only=True)
    (root/'runs/track4world-complete-schema-audit-001.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)
    if report['missing_nonmetric'] or report['unexpected'] or report['shape_mismatch'] or missing!=metric:
        raise ValueError('Geometry checkpoint/model structural mismatch')


if __name__=='__main__':main()
