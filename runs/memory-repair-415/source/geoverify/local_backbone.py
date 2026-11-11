"""Scoped local weight routing; leave upstream architecture/forward untouched."""
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from threading import RLock

_LOAD_LOCK=RLock()


@contextmanager
def checkpoint_convnext_initializer(module):
    """Skip disposable ImageNet initialization before strict Track4World load.

The caller must reject every missing non-metric parameter after loading the
official tracking checkpoint; random constructor values must never run inference.
Only the upstream model module's ConvNeXt binding is changed temporarily.
"""
    with _LOAD_LOCK:
        original=module.ConvNeXt
        def construct(*args,**kwargs):
            kwargs['init_weights']=False
            return original(*args,**kwargs)
        module.ConvNeXt=construct
        try:
            yield
        finally:
            module.ConvNeXt=original


@contextmanager
def local_verifier_encoder(module,state):
    """Use the full Verifier checkpoint's CNN during upstream construction.

    Upstream initializes this CNN from a Hub CoTracker model before loading the
    full Verifier checkpoint. Reuse the exact saved CNN instead, requiring all
    keys/shapes. Only this module's torch binding is routed; global Hub is intact.
    """
    prefix='cnn_encoder.'
    encoder_state={key[len(prefix):]:value for key,value in state.items() if key.startswith(prefix)}
    encoder=module.BasicEncoder(input_dim=3,output_dim=128,stride=4)
    encoder.load_state_dict(encoder_state,strict=True)
    with _LOAD_LOCK:
        original=module.torch
        def load(repository,model,*args,**kwargs):
            if repository!='facebookresearch/co-tracker' or model!='cotracker3_offline' or args or kwargs:
                raise ValueError('Unexpected Verifier initialization Hub request')
            return SimpleNamespace(model=SimpleNamespace(fnet=encoder))
        class TorchBinding:
            hub=SimpleNamespace(load=load)
            def __getattr__(self,name):
                return getattr(original,name)
        module.torch=TorchBinding()
        try:
            yield
        finally:
            module.torch=original


@contextmanager
def local_hub_model(module,attribute,directory,expected_repository):
    """Route a PyTorchModelHubMixin class binding to a verified local folder."""
    directory=Path(directory).resolve()
    if not (directory/'config.json').is_file() or not (directory/'model.safetensors').is_file():
        raise FileNotFoundError('Complete local hub model required: '+str(directory))
    with _LOAD_LOCK:
        original=getattr(module,attribute)
        def load(repository,*args,**kwargs):
            if repository!=expected_repository:
                raise ValueError('Unexpected hub backbone requested: '+str(repository))
            kwargs['local_files_only']=True
            metric_only=(directory/'metric_source.json').is_file()
            if metric_only:kwargs['strict']=False
            model=original.from_pretrained(str(directory),*args,**kwargs)
            if metric_only:
                import torch
                from safetensors.torch import load_file
                selected=load_file(str(directory/'model.safetensors'))
                expected={k:v for k,v in model.state_dict().items() if k.startswith('model.da3_metric.')}
                if set(selected)!=set(expected) or not expected:
                    raise ValueError('Metric-only initialization must cover every metric parameter')
                if any(not torch.equal(expected[k].cpu(),value) for k,value in selected.items()):
                    raise ValueError('Metric branch did not load the exact selected tensors')
            return model
        setattr(module,attribute,SimpleNamespace(from_pretrained=load))
        try:
            yield
        finally:
            setattr(module,attribute,original)


@contextmanager
def local_auto_model(module,directory,expected_repository):
    directory=Path(directory).resolve()
    if not (directory/'config.json').is_file() or not (directory/'model.safetensors').is_file():
        raise FileNotFoundError('Complete local backbone config and safetensors required: '+str(directory))
    # Patch only the upstream adapter module's binding, not transformers globally.
    # All calls through this helper serialize model construction and restore on error.
    with _LOAD_LOCK:
        original=module.AutoModel
        def load(repository,*args,**kwargs):
            if repository!=expected_repository:
                raise ValueError('Unexpected backbone requested: '+str(repository))
            kwargs['local_files_only']=True
            kwargs['trust_remote_code']=False
            kwargs['use_safetensors']=True
            return original.from_pretrained(str(directory),*args,**kwargs)
        module.AutoModel=SimpleNamespace(from_pretrained=load)
        try:
            yield
        finally:
            module.AutoModel=original
