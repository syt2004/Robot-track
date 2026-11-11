"""Persistent isolated Track4World process; mmap frames, return sparse results."""
import argparse
import json
from pathlib import Path
import time
import traceback
import numpy as np


def atomic_json(path, value):
    temporary = path.with_suffix('.pending')
    temporary.write_text(json.dumps(value))
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', required=True)
    parser.add_argument('--root', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--backbone-directory')
    args = parser.parse_args()
    directory = Path(args.directory).resolve()
    try:
        from geoverify.track4world_adapter import Track4WorldAdapter
        model = Track4WorldAdapter(args.root,args.checkpoint,backbone_directory=args.backbone_directory)
        atomic_json(directory/'ready.json',dict(ready=True))
        last_job = None
        while not (directory/'stop').exists():
            request_path = directory/'request.json'
            if not request_path.exists():
                time.sleep(0.01)
                continue
            request = json.loads(request_path.read_text())
            job = request['job_id']
            if job == last_job:
                time.sleep(0.01)
                continue
            rgb = np.load(directory/'frames.npy',mmap_mode='r',allow_pickle=False)
            queries = np.load(directory/'queries.npy',allow_pickle=False)
            from geoverify.correlation_pyramid import complete_correlation_pyramid
            from track4world.nets import blocks
            with complete_correlation_pyramid(blocks):
                result = model.run_window(rgb,queries)
            result['correlation_pyramid_protocol'] = np.asarray('skip_unused_final_downsample_v1')
            np.savez(directory/'result.pending.npz',**result)
            (directory/'result.pending.npz').replace(directory/'result.npz')
            atomic_json(directory/'response.json',dict(
                job_id=job,completed_monotonic=time.monotonic(),
                anchor_frame=request['anchor_frame'],target_frame=request['target_frame']))
            last_job = job
    except Exception:
        atomic_json(directory/'error.json',dict(error=traceback.format_exc()))
        raise


if __name__ == '__main__':
    main()
