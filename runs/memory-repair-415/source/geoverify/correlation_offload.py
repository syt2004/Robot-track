"""Inference-only CPU storage of full correlations; all math stays on original device."""
import ast,inspect,textwrap
from contextlib import contextmanager

def patch_initializer(original,store):
 tree=ast.parse(textwrap.dedent(inspect.getsource(original)));appended=0;downsampled=0
 for node in ast.walk(tree):
  if not isinstance(node,ast.For) or ast.unparse(node.iter)!='range(self.num_levels)':continue
  body=[]
  for s in node.body:
   if isinstance(s,ast.Assign) and ast.unparse(s.targets[0])=='fmap2' and isinstance(s.value,ast.Call) and ast.unparse(s.value.func)=='F.interpolate':
    guard=ast.parse('if i+1 < self.num_levels:\n pass').body[0];guard.body=[s];body.append(guard);downsampled+=1
   elif isinstance(s,ast.Expr) and isinstance(s.value,ast.Call) and ast.unparse(s.value.func)=='self.corr_pyramid.append':
    assert ast.unparse(s.value.args[0])=='corr';s.value.args[0]=ast.parse('_offload_store(corr)',mode='eval').body;body.extend([s,ast.parse('del corr').body[0]]);appended+=1
   else:body.append(s)
  node.body=body
 if (appended,downsampled)!=(1,1):raise RuntimeError('Unexpected original correlation initializer')
 ast.fix_missing_locations(tree);ns={**original.__globals__,'_offload_store':store};exec(compile(tree,'<correlation_cpu_storage>','exec'),ns);return ns[original.__name__]
def patch_lookup(original,load):
 tree=ast.parse(textwrap.dedent(inspect.getsource(original)));changed=0
 for node in ast.walk(tree):
  if isinstance(node,ast.Assign) and ast.unparse(node.targets[0])=='corr' and ast.unparse(node.value)=='self.corr_pyramid[i]':node.value=ast.parse('_offload_load(self.corr_pyramid[i],coords.device)',mode='eval').body;changed+=1
 if changed!=1:raise RuntimeError('Unexpected original correlation lookup')
 ast.fix_missing_locations(tree);ns={**original.__globals__,'_offload_load':load};exec(compile(tree,'<correlation_cpu_restore>','exec'),ns);return ns[original.__name__]
@contextmanager
def offloaded_correlations(blocks):
 original_init=blocks.CorrBlock.__init__;original_call=blocks.CorrBlock.__call__;stats=dict(stored_tensors=0,stored_bytes=0,loaded_tensors=0,loaded_bytes=0,max_tensor_bytes=0)
 def store(tensor):
  if tensor.requires_grad:raise ValueError('Offloaded correlation protocol is inference only')
  size=tensor.numel()*tensor.element_size();stats['stored_tensors']+=1;stats['stored_bytes']+=size;stats['max_tensor_bytes']=max(stats['max_tensor_bytes'],size);return tensor.to(device='cpu')
 def load(tensor,device):
  if tensor.device.type!='cpu':raise ValueError('Expected correlation stored on CPU')
  stats['loaded_tensors']+=1;stats['loaded_bytes']+=tensor.numel()*tensor.element_size();return tensor.to(device=device)
 try:
  blocks.CorrBlock.__init__=patch_initializer(original_init,store);blocks.CorrBlock.__call__=patch_lookup(original_call,load);yield stats
 finally:blocks.CorrBlock.__init__=original_init;blocks.CorrBlock.__call__=original_call
