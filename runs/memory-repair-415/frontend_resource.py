"""Frozen model with verified RGB buffers and process-local NumPy allocation policy."""
from pathlib import Path
import os,sys,json,hashlib,runpy,faulthandler
assert os.environ.get('NUMPY_MADVISE_HUGEPAGE')=='0'
import numpy as np
assert not np.core.multiarray._get_madvise_hugepage()
O=Path(__file__).resolve().parent;sys.path.insert(0,str(O));case=Path(sys.argv[sys.argv.index('--output')+1]).parent
from verifier_transport_v2 import install
install();faulthandler.enable()
(case/'frontend_observer.json').write_text(json.dumps(dict(scope='Verified pinned uint8/float RGB buffers; process-local NumPy hugepage advice disabled; no recurring asynchronous traceback sampler',files={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),O/'verifier_transport_v2.py',O/'frontend.py']},numpy_madvise_hugepage=np.core.multiarray._get_madvise_hugepage()),indent=2))
runpy.run_path(str(O/'frontend.py'),run_name='__main__')
