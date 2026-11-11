import numpy as np
from .cache import validate_arrays
from .calibration import DESCRIPTOR_DIM,DESCRIPTOR_PROTOCOL
from .feature_schema import FEATURE_NAMES

def validate_appearance_arrays(cache):
 a=validate_arrays(cache,len(FEATURE_NAMES))
 if 'appearance_descriptor' not in cache:raise ValueError('Missing appearance descriptor')
 descriptor=np.asarray(cache['appearance_descriptor'])
 if descriptor.shape!=(*a['valid'].shape,DESCRIPTOR_DIM) or descriptor.dtype.kind!='f':raise ValueError('Appearance descriptor schema mismatch')
 if not np.isfinite(descriptor).all() or np.any(descriptor[~a['valid']]!=0):raise ValueError('Invalid visual descriptor or unmasked unavailable candidate')
 a['appearance_descriptor']=descriptor
 return a

def validate_descriptor_protocol(value):
 if not isinstance(value,dict) or value.get('version')!=DESCRIPTOR_PROTOCOL or value.get('dimension')!=DESCRIPTOR_DIM or value.get('dtype')!='float32' or value.get('masked_invalid')!='zero':raise ValueError('Unsupported descriptor protocol')
 for key in ['implementation_sha256','provider_weights_sha256']:
  v=value.get(key,'')
  if len(v)!=64 or any(x not in '0123456789abcdef' for x in v):raise ValueError('Missing descriptor identity')
 return value
