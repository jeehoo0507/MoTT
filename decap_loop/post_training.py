"""Three distinct RL objectives/start distributions; certified training archive only."""
import hashlib
import numpy as np
import torch
from .reserve_curriculum import ReserveCurriculum

ARMS=('pt_control','pt_feasible','pt_robust','pt_full')


class PostTraining:
    def __init__(self,arm,phase,hours):
        if arm not in ARMS or phase not in (1,2,3):raise ValueError('Invalid post-training arm/phase')
        self.arm=arm;self.phase=phase;self.feedback=ReserveCurriculum('adaptive',hours)
        self.archive={};self._keys={};self.iteration=0
        self.stats=dict(certified_candidates=0,archive_improvements=0,refinement_episodes=0,construction_episodes=0,teacher_checks=0)

    @property
    def stage(self):
        if self.arm=='pt_control':return 0
        if self.arm=='pt_feasible':return 1
        return min(self.phase,2) if self.arm=='pt_robust' else self.phase

    @property
    def count_weight(self):return {0:1.,1:.1,2:.3,3:1.}[self.stage]

    def reserve(self,batch):
        if self.stage!=2 or self.phase!=2:return 0.
        return max(0.,1-self.feedback.elapsed/self.feedback.seconds)*self.feedback.gaps.get(self.feedback.key(batch),.1)

    def key(self,batch):
        if id(batch) not in self._keys:
            h=hashlib.sha256(self.feedback.key(batch).encode())
            for x in (batch.valid,batch.keepout,batch.entry,batch.probes):h.update(x.cpu().numpy().tobytes())
            h.update(batch.zt_par.tobytes());self._keys[id(batch)]=h.hexdigest()
        return self._keys[id(batch)]

    def add(self,batch,indices,occ,exact):
        bank=self.archive.setdefault(self.key(batch),{})
        for i,placement,worst in zip(indices.cpu().tolist(),occ.cpu().tolist(),exact):
            if not np.isfinite(worst):raise ValueError('Nonfinite archive certificate')
            if worst>1e-3:continue
            self.stats['certified_candidates']+=1
            count=sum(t>=0 for t in placement);previous=bank.get(str(i))
            if previous is None or count<previous['count']:
                bank[str(i)]=dict(occ=placement,count=count,exact_margin=float(worst));self.stats['archive_improvements']+=1

    def wants_refinement(self,i):return self.stage==3 and (self.iteration+int(i))%2==0

    def initial(self,batch,indices):
        occ=torch.full((len(indices),batch.n),-1,dtype=torch.long,device=indices.device)
        bank=self.archive.get(self.key(batch),{})
        for e,i in enumerate(indices.cpu().tolist()):
            row=bank.get(str(i))
            if self.wants_refinement(i) and row:
                occ[e]=torch.tensor(row['occ'],device=indices.device);self.stats['refinement_episodes']+=1
            else:self.stats['construction_episodes']+=1
        return occ

    def reward(self,cnt,fail,out,reference,lam):
        start=out.get('start_count',torch.zeros(cnt.numel(),device=cnt.device)).view_as(cnt)
        # Same initial placement within each M group: offset cannot favor archive over construction samples.
        gain=(start-cnt) if self.stage==3 else -cnt
        return self.count_weight*gain/reference[:,None]-lam*(fail.float()+out['V'].float().view_as(cnt))

    def state(self):return dict(arm=self.arm,phase=self.phase,feedback=self.feedback.state(),archive=self.archive,stats=self.stats,iteration=self.iteration)

    def restore(self,state,transfer=False):
        if not state:return
        if state['arm']!=self.arm or (state['phase']!=self.phase and not (transfer and state['phase']==self.phase-1)):
            raise ValueError('Wrong archive arm/phase')
        self.archive=state['archive'];self.stats=state['stats']
        if transfer:
            self.feedback.gaps=state['feedback']['gaps'];self.feedback.observations=state['feedback']['observations']
        else:self.feedback.restore(state['feedback']);self.iteration=state['iteration']
