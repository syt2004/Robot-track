"""Skip CorrBlock's unused final downsample, preserving every sampled level."""
import ast
from contextlib import contextmanager
import inspect
import textwrap

def _patched_initializer(original):
    tree=ast.parse(textwrap.dedent(inspect.getsource(original)))
    changed=0
    for node in ast.walk(tree):
        if not isinstance(node,ast.For) or not isinstance(node.target,ast.Name) or node.target.id!='i':
            continue
        if ast.unparse(node.iter)!='range(self.num_levels)':
            continue
        body=[]
        for statement in node.body:
            match=(isinstance(statement,ast.Assign) and len(statement.targets)==1
                   and isinstance(statement.targets[0],ast.Name) and statement.targets[0].id=='fmap2'
                   and isinstance(statement.value,ast.Call)
                   and ast.unparse(statement.value.func)=='F.interpolate')
            if match:
                guard=ast.parse('if i + 1 < self.num_levels:\n    pass').body[0]
                guard.body=[statement];body.append(guard);changed+=1
            else:
                body.append(statement)
        node.body=body
    if changed!=1:
        raise RuntimeError('Unsupported CorrBlock initializer; expected one pyramid downsample')
    ast.fix_missing_locations(tree);namespace={}
    exec(compile(tree,'<corr_pyramid_skip_unused_final_v1>','exec'),original.__globals__,namespace)
    patched=namespace[original.__name__]
    def checked(self,fmap1,fmap2,corr_levels,corr_radius,mode='area',save_fmap=False):
        if corr_levels<1 or min(fmap2.shape[-2:])<2**(corr_levels-1):
            raise ValueError('Feature map too small for all requested correlation levels')
        return patched(self,fmap1,fmap2,corr_levels,corr_radius,mode=mode,save_fmap=save_fmap)
    return checked

@contextmanager
def complete_correlation_pyramid(blocks):
    """Worker-local patch; no resize, dropped level, weight change or fallback."""
    original=blocks.CorrBlock.__init__
    blocks.CorrBlock.__init__=_patched_initializer(original)
    try:
        yield
    finally:
        blocks.CorrBlock.__init__=original
