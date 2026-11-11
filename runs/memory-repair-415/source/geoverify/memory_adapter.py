"""Online delayed geometry repair; RGB and arrived model predictions only."""
from collections import OrderedDict
import json
from pathlib import Path
import numpy as np
import torch
from .memory_state import MemoryLedger,MemoryEdit
from .memory_repair import sample_features, bound_to_reference

class RepairingTrackOn:
    def __init__(self,base,repair,history_frames=24,audit_path=None):
        self.base,self.repair=base,repair
        self.history_frames=history_frames
        self.audit_path=Path(audit_path) if audit_path else None
        self.verifier=None
        self.cohort=-1
        self.reset()
    def reset(self):
        self.cohort+=1
        self.base.reset();self.ledger=None;self.frames=OrderedDict();self.frame_id=0
        self.rgb_frames=OrderedDict()
        self.events=[];self.reference_memory=None;self.shadow_memory=None;self.shadow_mask=None;self.shadow_anchor=None
    def step(self,rgb,new_queries=None):
        model=self.base.predictor.model;extract=model.extract_frame_features;captured={}
        def capture(frame):
            features=extract(frame);captured['all']=features
            captured['fused']=features[-1].detach().to(device='cpu',dtype=torch.float32)
            return features
        model.extract_frame_features=capture
        try:raw=self.base.step(rgb,new_queries=new_queries)
        finally:model.extract_frame_features=extract
        p=self.base.predictor;current=p.point_memory[:p.N]
        max_transition=0.;clipped_queries=0
        with torch.inference_mode():
            if self.shadow_memory is None:
                # Before the first evidence arrival both streams are identical.
                self.shadow_memory=current.clone();self.shadow_mask=p.temporal_mask[:p.N].clone()
                self.shadow_anchor=p.q_init[:p.N].clone();shadow_xy=np.asarray(raw['xy']).copy();shadow_visibility=np.asarray(raw['visibility']).copy()
            else:
                if len(self.shadow_memory)!=p.N or not torch.equal(self.shadow_anchor,p.q_init[:p.N]):
                    raise ValueError('Each query cohort must retain its fixed query identities and anchors')
                # A second decoder pass shares the exact RGB features, weights,
                # and original query anchors, but never consumes repaired state.
                shadow_xy,shadow_logit,shadow_new=model.track_frame(self.shadow_anchor,self.shadow_mask,
                    self.shadow_memory,captured['all'],rgb.shape[0],rgb.shape[1])
                self.shadow_memory=torch.cat((self.shadow_memory[:,1:],shadow_new[:,None]),1)
                self.shadow_mask=torch.cat((self.shadow_mask[:,1:],torch.zeros(p.N,1,dtype=torch.bool,device=current.device)),1)
                transition=(current[:,-1]-shadow_new).norm(dim=-1)/shadow_new.norm(dim=-1).clamp_min(1e-6)
                max_transition=float(transition.max());clipped_queries=int((transition>self.repair.maximum_relative_change).sum())
                current[:,-1]=bound_to_reference(current[:,-1],shadow_new,self.repair.maximum_relative_change)
                shadow_xy=shadow_xy.cpu().numpy();shadow_visibility=shadow_logit.sigmoid().cpu().numpy()
            self.reference_memory=self.shadow_memory.clone()
            relative=(current-self.reference_memory).norm(dim=-1)/self.reference_memory.norm(dim=-1).clamp_min(1e-6)
            valid=~p.temporal_mask[:p.N];maximum=float(relative[valid].max()) if valid.any() else 0.
            if not torch.isfinite(current).all() or maximum>self.repair.maximum_relative_change+1e-5:
                raise ValueError('Memory left the independent shadow-state bound')
        if self.ledger is None:self.ledger=MemoryLedger(np.arange(len(raw['xy'])),model.M)
        self.ledger.append(self.frame_id,float(self.frame_id),shadow_xy,shadow_visibility)
        self.frames[self.frame_id]=(captured['fused'],rgb.shape[0],rgb.shape[1])
        self.rgb_frames[self.frame_id]=np.asarray(rgb).copy()
        while len(self.frames)>self.history_frames:self.frames.popitem(last=False)
        while len(self.rgb_frames)>self.history_frames:self.rgb_frames.popitem(last=False)
        self.shadow_xy=shadow_xy.copy();self.shadow_visibility=shadow_visibility.copy()
        if self.audit_path is not None:
            self.audit_path.parent.mkdir(parents=True,exist_ok=True)
            record=dict(cohort=self.cohort,frame=self.frame_id,max_shadow_relative_change=maximum,max_unclipped_transition_relative_change=max_transition,transition_clipped_queries=clipped_queries,
                mean_shadow_xy_difference=float(np.linalg.norm(np.asarray(raw['xy'])-shadow_xy,axis=-1).mean()),
                reference='independent unmodified Track-On state; shared RGB encoder, separate decoder state')
            with self.audit_path.with_name('state_bounds.jsonl').open('a') as stream:stream.write(json.dumps(record)+'\n')
        self.frame_id+=1
        raw['source']=raw.get('source','trackon')+'+verified_geometry_shadow_memory_v4'
        return raw
    def apply_geometry(self,result,now):
        """Apply after this frame's prediction, for use by subsequent frames."""
        if result.completed_timestamp>now:return
        p=self.base.predictor;m=p.model;ledger=self.ledger
        if ledger is None:return
        a=result.arrays;count=len(a['xy_projected'])
        if count!=result.target_frame-result.anchor_frame+1:
            raise ValueError('Repair requires explicit consecutive window frame IDs')
        # Ledger time uses frame indices internally; availability is checked above.
        # Slot location must not compare wall-clock seconds against frame indices.
        if self.verifier is None:raise RuntimeError('Historical write verification is required')
        ids=np.asarray(result.query_ids,dtype=int)
        if ((ids<0)|(ids>=p.N)).any():raise ValueError('Geometry query identity mismatch')
        window=list(range(result.anchor_frame,result.target_frame+1))
        present=all(ft in self.rgb_frames and np.count_nonzero(ledger.frames==ft)==1 for ft in window)
        verified=None
        if present:
            from .verified_write import verify_historical_window
            reference=np.stack([ledger.xy[ids,int(np.flatnonzero(ledger.frames==ft)[0])] for ft in window])
            verified=verify_historical_window(self.verifier,np.stack([self.rgb_frames[ft] for ft in window]),
                reference,np.asarray(a['xy_projected']),np.asarray(a['valid_3d'],bool))
        edits=[]
        with torch.inference_mode():
            memory=p.point_memory[:p.N];sampled=torch.zeros_like(memory)
            evidence=memory.new_zeros((*memory.shape[:2],8));valid=torch.zeros_like(p.temporal_mask[:p.N])
            for j,ft in enumerate(range(result.anchor_frame,result.target_frame+1)):
                if ft not in self.frames or verified is None:continue
                slots=np.flatnonzero(ledger.frames==ft)
                if not len(slots):continue
                slot=int(slots[0]);fused,h,w=self.frames[ft]
                ids=np.asarray(result.query_ids,dtype=int)
                if ((ids<0)|(ids>=p.N)).any():raise ValueError('Geometry query identity mismatch')
                coords=np.asarray(a['xy_projected'][j]);ok=np.asarray(a['valid_3d'][j],bool)&np.isfinite(coords).all(-1)
                ok &= np.array([(result.job_id,int(q)) not in ledger.edited[slot] for q in ids])
                ok &= (coords[:,0]>=0)&(coords[:,0]<w)&(coords[:,1]>=0)&(coords[:,1]<h)
                old=ledger.xy[ids,slot];safe=np.where(ok[:,None],coords,old)
                xy=torch.tensor(safe,device=memory.device,dtype=memory.dtype)
                sampled[ids,slot]=sample_features(m,fused.to(memory.device),xy,h,w)
                delta=(safe-old)*np.array([256/w,256/h]);age=self.frame_id-ft
                ev=np.c_[delta/16,np.linalg.norm(delta,axis=-1)/16,a['visconf'][j],np.full(len(ids),age/72),np.full(len(ids),(self.frame_id-result.target_frame-1)/24),np.ones(len(ids)),np.zeros(len(ids))]
                ev[:,-2:]=verified['features'][j]
                ok &= verified['accept'][j]
                ok &= np.isfinite(ev).all(-1)
                evidence[ids,slot]=torch.tensor(np.nan_to_num(ev),device=memory.device,dtype=memory.dtype)
                valid[ids,slot]=torch.tensor(ok,device=memory.device)
                edits.extend((int(q),slot) for q in ids[ok])
            proposal=self.repair(self.reference_memory,sampled,evidence,valid)
            # Each proposal is anchored to the unmodified shadow trajectory; it must not
            # accumulate another residual onto an already repaired slot.
            updated=torch.where(valid[...,None],proposal,memory)
            if not torch.isfinite(updated).all():raise ValueError('Nonfinite repaired memory')
            difference=float((updated-memory).abs().max())
            relative=(updated-self.reference_memory).norm(dim=-1)/self.reference_memory.norm(dim=-1).clamp_min(1e-6)
            active=~p.temporal_mask[:p.N]
            max_shadow=float(relative[active].max()) if active.any() else 0.
            if max_shadow>self.repair.maximum_relative_change+1e-5:raise ValueError('Geometry update left shadow-state bound')
            p.point_memory[:p.N]=updated
            for q,slot in edits:ledger.edited[slot].add((result.job_id,q))
        self.events.append(dict(job_id=result.job_id,effective_frame=self.frame_id,slots=len(edits),max_change=difference,max_shadow_relative_change=max_shadow,reference='independent_shadow',verification_window_available=present,
            geometry_valid_slots=int(np.asarray(a['valid_3d'],bool).sum()),
            verified_slots=int(verified['accept'].sum()) if verified is not None else 0,
            minimum_verified_margin=float((verified['scores'][...,1]-verified['scores'][...,0])[verified['accept']].min()) if verified is not None and verified['accept'].any() else None,
            verification_protocol='paired_historical_verifier_margin_v1'))
        if self.audit_path is not None:
            self.audit_path.parent.mkdir(parents=True,exist_ok=True)
            event=dict(self.events[-1],cohort=self.cohort,coordinate_scope='cohort-local query and frame indices',geometry_anchor_frame=result.anchor_frame,
                       geometry_target_frame=result.target_frame,completed_timestamp=result.completed_timestamp,
                       applied_timestamp=now,received_frame=self.frame_id-1,
                       query_ids=sorted({q for q,slot in edits}))
            with self.audit_path.open('a') as stream:stream.write(json.dumps(event)+'\n')
        self.events=self.events[-128:]

class MemoryGeometryWorker:
    def __init__(self,worker,tracker):self.worker,self.tracker=worker,tracker
    def __getattr__(self,name):return getattr(self.worker,name)
    def poll(self,current_frame,now):
        result=self.worker.poll(current_frame,now)
        if result is not None:self.tracker.apply_geometry(result,now)
        return result
