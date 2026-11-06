"""Exploratory temporal memory MoTT; delta-rule inspired, not an official DeltaNet replica.

The memory axis is placement time, never arbitrary candidate-cell order. Exact action effects
and safety masks are retained. Caching only reduces neural inference work, not physics work.
"""
import torch
from torch import nn
from torch.nn import functional as F

from .model import MoTT
from .sim import K_MODES
from .peak_env import mott_pk_dims

ARMS = ('mott', 'cached', 'gru', 'delta', 'gru_aux', 'delta_aux')


def detach_state(state):
    return {k: v.detach() for k, v in state.items()}


class MemoryMoTT(MoTT):
    def __init__(self, arm, n_types=3, d=64, layers=3, heads=4):
        if arm not in ARMS:
            raise ValueError(arm)
        super().__init__(n_types, K_MODES, mott_pk_dims(n_types), d=d, layers=layers, heads=heads)
        self.arm, self.d = arm, d
        self.kind = arm.split('_')[0]
        self.use_aux = arm.endswith('_aux')
        if self.kind in ('gru', 'delta'):
            self.write = nn.Linear(3*d, 2*d + 2)
            self.read = nn.Linear(d, d)
            self.memory_out = nn.Linear(d, d, bias=False)
        if self.kind == 'gru':
            # One shared GRU cell per memory row; same d*d state capacity as delta memory.
            self.cell = nn.GRUCell(d, d)
        if self.use_aux:
            self.phys_head = nn.Sequential(nn.Linear(2*d, d), nn.GELU(), nn.Linear(d, 16))

    def unpack(self, x):
        E, n, _ = x.shape
        o = self.dn
        node = x[..., :o]
        edge = x[..., o:o+self.K*self.de].reshape(E, n, self.K, self.de); o += self.K*self.de
        mode = x[:, 0, o:o+self.K*self.dm].reshape(E, self.K, self.dm); o += self.K*self.dm
        dec = x[:, 0, o:o+self.T*self.dc].reshape(E, self.T, self.dc)
        mmask = mode[..., 3] > .5
        mmask = mmask | ~mmask.any(1, keepdim=True)
        return node, edge, mode, dec, mmask

    def encode(self, x, valid):
        node, edge, mode, dec, mmask = self.unpack(x)
        u, m, c = self.embed_u(node), self.embed_m(mode), self.embed_c(dec)
        bias = self.edge_bias(edge).permute(0, 3, 1, 2)
        ev = self.edge_val(edge)
        for blk in self.blocks:
            m, u, c = blk(m, u, c, bias, ev, mmask, valid, self.fc_bias(mode, dec))
        return u, m, c

    def decode(self, node, u, m, c, valid, mmask):
        E, n, d = u.shape
        uu, cc = u[:, :, None].expand(E,n,self.T,d), c[:, None].expand(E,n,self.T,d)
        add = self.add_head(torch.cat((uu,cc,uu*cc), -1))[..., 0]
        nb = 4 + self.T + 3
        dv, mn = node[...,nb:nb+self.T+1], node[...,nb+self.T+1:nb+2*(self.T+1)]
        logits = torch.cat((add,self.rm_head(u)), -1) + self.w_dv*dv - self.w_mn*mn
        vm = valid[...,None].to(u.dtype)
        g = (u*vm).sum(1)/vm.sum(1).clamp(min=1)
        mg = (m*mmask[...,None]).sum(1)/mmask.sum(1,keepdim=True).clamp(min=1)
        return logits, self.stop_head(torch.cat((g,mg), -1))[:,0], g

    def step(self, x, valid, state=None):
        node, edge, mode, dec, mmask = self.unpack(x)
        E, n, _ = x.shape
        if self.kind == 'mott':
            u,m,c = self.encode(x, valid)
            new = {}
        else:
            if state is None:
                u0,m0,c0 = self.encode(x, valid)
                state = dict(u0=u0,m0=m0,c0=c0,prev=u0.new_zeros(E,self.d),
                             action=u0.new_zeros(E,self.d),memory=u0.new_zeros(E,self.d,self.d))
            cm = self.embed_m(mode)
            mp = (cm*mmask[...,None]).sum(1)/mmask.sum(1,keepdim=True).clamp(min=1)
            # Current observations remain available to every cached arm, including the no-history control.
            ep = (self.edge_val(edge)*mmask[:,None,:,None]).sum(2)/mmask.sum(1)[:,None,None].clamp(min=1)
            u = state['u0'] + self.embed_u(node) + ep + mp[:,None]
            m,c = state['m0'] + cm, state['c0'] + self.embed_c(dec)
            vm = valid[...,None].to(u.dtype)
            ctx = (u*vm).sum(1)/vm.sum(1).clamp(min=1)
            memory = state['memory']
            if self.kind in ('gru','delta'):
                z = self.write(torch.cat((ctx, ctx-state['prev'], state['action']), -1))
                k,v = F.normalize(z[:,:self.d],dim=-1), torch.tanh(z[:,self.d:2*self.d])
                alpha,beta = torch.sigmoid(z[:,-2:-1]+3), torch.sigmoid(z[:,-1:])
                if self.kind == 'delta':
                    decayed = alpha[...,None]*memory
                    error = v - torch.einsum('bi,bij->bj',k,decayed)
                    memory = decayed + beta[...,None]*k[:,:,None]*error[:,None,:]
                else:
                    inp = k[:,:,None]*v[:,None,:]
                    memory = self.cell(inp.reshape(-1,self.d),memory.reshape(-1,self.d)).reshape(E,self.d,self.d)
                q = F.normalize(self.read(u),dim=-1)
                u = u + self.memory_out(q @ memory)
            new = {**state,'prev':ctx,'memory':memory}
        logits,stop,g = self.decode(node,u,m,c,valid,mmask)
        new.update(u=u,c=c,g=g)
        return torch.cat((logits.reshape(E,-1),stop[:,None]),1), new

    def action_embedding(self, state, action):
        E,n,d = state['u'].shape
        stop = action == n*(self.T+1)
        cell = (action//(self.T+1)).clamp(max=n-1)
        ch = action%(self.T+1)
        rows = torch.arange(E,device=action.device)
        # Stop and removal do not spuriously inherit a decap-type embedding.
        chosen = state['u'][rows,cell] + state['c'][rows,ch.clamp(max=self.T-1)]*(ch<self.T)[:,None]
        return torch.where(stop[:,None],state['g'],chosen)

    def advance(self, state, action):
        return {**state, 'action':self.action_embedding(state,action)}

    def physics(self, state, action):
        return self.phys_head(torch.cat((state['g'], self.action_embedding(state,action)), -1))
