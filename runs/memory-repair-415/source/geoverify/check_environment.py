"""Read-only real import/CUDA probe; does not load weights or claim model quality.

Run separately in each isolated model environment. No mocks or fallback modules.
"""
import argparse
import importlib
from importlib.metadata import version
import json
from pathlib import Path
import sys
import traceback


def check(component,root,require_cuda=True):
    report=dict(component=component,python=sys.version,executable=sys.executable,
                state='checking',weights_loaded=False,versions={},cuda_required=require_cuda)
    try:
        expected={'trackon':('3.12','2.4.1+cu121','0.19.1+cu121','4.56.1'),
                  'track4world':('3.11','2.5.1+cu121','0.20.1+cu121','4.57.3')}[component]
        if '%d.%d'%sys.version_info[:2]!=expected[0]:
            raise RuntimeError('Wrong Python version; expected '+expected[0])
        for package,required in zip(('torch','torchvision','transformers','numpy'),expected[1:]+('1.26.4',)):
            actual=version(package)
            report['versions'][package]=actual
            if actual!=required:
                raise RuntimeError('%s version %s differs from required %s'%(package,actual,required))
        torch=importlib.import_module('torch')
        importlib.import_module('torchvision')
        sys.path.insert(0,str(Path(root).resolve(strict=True)))
        if component=='trackon':
            report['versions']['mmcv']=version('mmcv')
            if report['versions']['mmcv']!='2.2.0':
                raise RuntimeError('Expected MMCV 2.2.0')
            importlib.import_module('mmcv.ops')
            importlib.import_module('model.trackon_predictor')
            importlib.import_module('verifier.verifier')
            importlib.import_module('verifier.verifier_predictor')
        else:
            importlib.import_module('track4world.nets.model')
        report['cuda_available']=torch.cuda.is_available()
        if require_cuda:
            if not report['cuda_available']:
                raise RuntimeError('CUDA is not available in this environment')
            probe=torch.arange(8.,device='cuda:0').square().sum()
            torch.cuda.synchronize()
            if probe.item()!=140.:
                raise RuntimeError('CUDA arithmetic probe failed')
            report['device']=torch.cuda.get_device_name(0)
        report['state']='imports_and_cuda_verified' if require_cuda else 'imports_verified_only'
    except Exception as error:
        report.update(state='failed',error=type(error).__name__+': '+str(error),traceback=traceback.format_exc())
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--component',choices=('trackon','track4world'),required=True)
    parser.add_argument('--root',required=True)
    parser.add_argument('--imports-only',action='store_true',help='Do not claim CUDA readiness')
    args=parser.parse_args()
    report=check(args.component,args.root,not args.imports_only)
    print(json.dumps(report,indent=2))
    if report['state']=='failed':
        raise SystemExit(1)


if __name__=='__main__':
    main()
