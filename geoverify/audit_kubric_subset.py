"""Annotation-only coverage audit; never a model performance visualization."""
import argparse
import json
from pathlib import Path
import numpy as np
from .datasets import file_sha256
from .kubric import iter_kubric


def audit(manifest_path, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    manifest_path=Path(manifest_path).resolve()
    manifest=json.loads(manifest_path.read_text())
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False)
    reports=[]
    for split in dict.fromkeys(s['split'] for s in manifest['scenes']):
        for sample in iter_kubric(manifest_path,split):
            query=sample.query_txy
            eligible=np.arange(len(sample.rgb))[:,None]>query[:,0]
            occluded=(~sample.target_visible)&eligible
            visible=sample.target_visible&eligible
            # Motion is measured only between successive visible annotations.
            valid_pairs=visible[1:]&sample.target_visible[:-1]
            motion=np.linalg.norm(np.diff(sample.target_xy,axis=0),axis=-1)
            per_point_motion=np.divide(np.where(valid_pairs,motion,0).sum(0),valid_pairs.sum(0),
                                       out=np.full(len(query),np.nan),where=valid_pairs.sum(0)>0)
            any_occluded=occluded.any(0)
            report=dict(scene=sample.name,split=split,points=len(query),frames=len(sample.rgb),
                        any_occlusion_points=int(any_occluded.sum()),
                        visible_fraction=float(visible.sum()/eligible.sum()),
                        median_visible_step_motion_pixels=float(np.nanmedian(per_point_motion)))
            reports.append(report)
            fig,axes=plt.subplots(1,3,figsize=(15,4.5),constrained_layout=True)
            axes[0].imshow(sample.rgb[0])
            initial=query[:,0]==0
            for mask,color,label in ((~any_occluded,'lime','Always visible after query'),
                                     (any_occluded,'magenta','Occluded at least once')):
                ids=initial&mask
                axes[0].scatter(query[ids,1],query[ids,2],s=10,c=color,label=label)
            axes[0].set_title('First-frame queries; colors from GT');axes[0].axis('off')
            axes[0].legend(fontsize=7,loc='lower left')
            axes[1].plot(np.arange(len(sample.rgb)),visible.sum(1),label='Visible')
            axes[1].plot(np.arange(len(sample.rgb)),occluded.sum(1),label='Occluded / outside image')
            axes[1].set(xlabel='Frame',ylabel='Evaluable query count',title='Annotation visibility over time')
            axes[1].legend(fontsize=8)
            axes[2].hist(per_point_motion[np.isfinite(per_point_motion)],bins=30)
            axes[2].set(xlabel='Mean visible step displacement (pixels)',ylabel='Query count',title='Image-space motion coverage')
            fig.suptitle('ANNOTATION AUDIT — Kubric '+sample.name+' / '+split+' — no model predictions')
            fig.savefig(output/(sample.name+'_coverage.png'),dpi=140);plt.close(fig)
    report=dict(dataset_manifest_sha256=file_sha256(manifest_path),model_predictions=False,
                limitation='Image-space displacement includes camera motion; this is not 3D object motion.',scenes=reports)
    (output/'coverage.json').write_text(json.dumps(report,indent=2)+'\n')
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-manifest',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();print(json.dumps(audit(args.dataset_manifest,args.output),indent=2))
