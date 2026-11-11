"""Deterministic sampled-frame near-duplicate screen, never proof of independence."""
import argparse, hashlib, json, os, sys
from pathlib import Path
import cv2
import numpy as np
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root))
from geoverify.datasets import NumpyDatasetUnpickler,robotap_record,file_sha256
from geoverify.geometry_worker import atomic_json
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
a.output.mkdir(parents=True,exist_ok=False)
state=dict(state='running',pid=os.getpid(),stage='decode');atomic_json(a.output/'status.json',state)
try:
    binding=root/'runs/robot-data-expansion-001/cross_member_groups.json'
    groups=json.loads(binding.read_text());rows=[];thumbs=[];hashes=[]
    membership={r['id']:i for i,g in enumerate(groups['groups']) for r in g['members']}
    for split in [0,4]:
        member=root/f'data/downloads/robotap_split{split}.pkl'
        invpath=root/f'runs/robot-data-expansion-001/split{split}_inventory/inventory.json'
        inv=json.loads(invpath.read_text());assert file_sha256(member)==inv['source_sha256']
        with member.open('rb') as stream:records=NumpyDatasetUnpickler(stream).load()
        for r in inv['videos']:
            s=robotap_record(r['name'],'unassigned',records[r['name']],24.,inv['source_sha256'])
            frames=np.unique(np.linspace(0,len(s.rgb)-1,16).astype(int));rgb=np.stack([cv2.resize(s.rgb[f],(64,64),interpolation=cv2.INTER_AREA) for f in frames])
            ph=[]
            for im in rgb:
                gray=cv2.resize(cv2.cvtColor(im,cv2.COLOR_RGB2GRAY),(32,32),interpolation=cv2.INTER_AREA).astype(np.float32)
                low=cv2.dct(gray)[:8,:8].flatten();bits=low>np.median(low[1:]);bits[0]=False;ph.append(bits)
            identity=inv['source_sha256']+':'+s.name
            rows.append(dict(id=identity,name=s.name,group=membership[identity],frames=frames.tolist(),source_sha256=inv['source_sha256']))
            thumbs.append(rgb);hashes.append(np.stack(ph));state.update(videos=len(rows),video=s.name);atomic_json(a.output/'status.json',state)
        del records
    pairs=[]
    for i in range(len(rows)):
        for j in range(i):
            if rows[i]['group']==rows[j]['group']:continue
            distances=np.count_nonzero(hashes[i][:,None]!=hashes[j][None,:],axis=-1)
            candidates=np.argwhere(distances<=8);best=None;matched=[]
            for x,y in candidates:
                mad=float(np.abs(thumbs[i][x].astype(float)-thumbs[j][y]).mean())
                if best is None or mad<best['rgb_mad']:best=dict(frame_i=int(rows[i]['frames'][x]),frame_j=int(rows[j]['frames'][y]),sample_i=int(x),sample_j=int(y),rgb_mad=mad,phash_distance=int(distances[x,y]))
                if mad<=10:matched.append((int(x),int(y)))
            if best is not None:
                ni=len({x for x,y in matched});nj=len({y for x,y in matched})
                pairs.append(dict(i=i,j=j,best=best,matched_frames_i=ni,matched_frames_j=nj,flag_for_review=min(ni,nj)>=3))
        state.update(stage='compare',compared_videos=i+1);atomic_json(a.output/'status.json',state)
    pairs.sort(key=lambda x:x['best']['rgb_mad'])
    atomic_json(a.output/'audit.json',dict(state='complete_screen',input_sha256=file_sha256(binding),script_sha256=file_sha256(__file__),videos=rows,pairs=pairs,flagged_pairs=sum(p['flag_for_review'] for p in pairs),protocol=dict(samples_per_video=16,phash_max_distance=8,rgb_mad_threshold=10,minimum_distinct_matches_each_side=3),limitation='Sampled same-view appearance screen, not exhaustive near-duplicate/cross-view/session verification. Flags require review, not automatic claims of leakage. No predictions or labels used to select videos.'))
    # Review the strongest cross-group similarities, including unflagged near misses.
    chosen=pairs[:24];canvas=np.full((max(1,len(chosen))*100,560,3),255,np.uint8)
    for k,pair in enumerate(chosen):
        i,j=pair['i'],pair['j'];b=pair['best'];y=k*100
        canvas[y:y+96,:96]=cv2.resize(thumbs[i][b['sample_i']],(96,96))
        canvas[y:y+96,100:196]=cv2.resize(thumbs[j][b['sample_j']],(96,96))
        for line,text in enumerate([rows[i]['name'][:48],rows[j]['name'][:48],f"MAD={b['rgb_mad']:.2f}; matches={pair['matched_frames_i']}/{pair['matched_frames_j']}; flag={pair['flag_for_review']}"]):
            cv2.putText(canvas,text,(202,y+22+line*24),cv2.FONT_HERSHEY_SIMPLEX,.36,(0,0,0),1)
    cv2.imwrite(str(a.output/'review.jpg'),cv2.cvtColor(canvas,cv2.COLOR_RGB2BGR))
    state.update(state='complete',stage='finished',flagged_pairs=sum(p['flag_for_review'] for p in pairs));atomic_json(a.output/'status.json',state)
except BaseException as e:
    state.update(state='failed',error=repr(e));atomic_json(a.output/'status.json',state);raise
