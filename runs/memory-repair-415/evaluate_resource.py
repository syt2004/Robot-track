"""Explicit resource policy plus persistent frontend exit-code evidence."""
from pathlib import Path
import os,sys,runpy,subprocess,json,signal,time
os.environ['NUMPY_MADVISE_HUGEPAGE']='0'
O=Path(__file__).resolve().parent;original=subprocess.Popen
class LoggedPopen(original):
 def __init__(self,command,*args,**kwargs):
  command=list(command);self._exit_case=None;self._exit_recorded=False
  if str(O/'frontend.py') in command:
   command=[str(O/'frontend_resource.py') if x==str(O/'frontend.py') else x for x in command]
   self._exit_case=Path(command[command.index('--output')+1]).parent
  super().__init__(command,*args,**kwargs)
 def record_exit(self,code):
  if code is None or self._exit_case is None or self._exit_recorded:return
  sig=signal.Signals(-code).name if code<0 else None
  (self._exit_case/'process_exit.json').write_text(json.dumps(dict(pid=self.pid,returncode=code,signal=sig,time=time.time()),indent=2));self._exit_recorded=True
 def poll(self):
  code=super().poll();self.record_exit(code);return code
 def wait(self,*args,**kwargs):
  code=super().wait(*args,**kwargs);self.record_exit(code);return code
subprocess.Popen=LoggedPopen
entry=O/('evaluate_pinned.py' if 'EVAL_GPU' in os.environ else 'evaluate.py')
sys.argv[0]=str(entry);runpy.run_path(str(entry),run_name='__main__')
