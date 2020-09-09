
#################################################
### THIS FILE WAS AUTOGENERATED! DO NOT EDIT! ###
#################################################
# file to edit: dev_nb/[Advanced] Hook.ipynb
import torch
from torch import nn
from functools import partial

class ForwardHook():
    def __init__(self, m, f):
        self.m = m
        self.hook = m.register_forward_hook(partial(f, self))
    @property
    def module(self): return self.m
    def remove(self): self.hook.remove()
    def __del__(self): self.remove()


class Hooks:
    def __init__(self, modules, hook_fn):
        self.handlers = []
        for m in modules:
            self.handlers.append(ForwardHook(m, hook_fn))

    def __enter__(self, *args): return self
    def __exit__ (self, *args): self.remove()
    def __del__(self): self.remove()
    def __iter__(self): return iter(self.handlers)
    def __getitem__(self, idx):
        if isinstance(idx, slice): return self.handlers[idx]
        if isinstance(idx[0],bool):
            assert len(idx)==len(self) # bool mask
            return [o for m,o in zip(idx,self.items) if m]
        return [self.items[i] for i in idx]
    def remove(self):
        for h in self.handlers:
            h.remove()

def get_hist(stats):
    return torch.stack(stats).T.float().log1p()

def get_min(stats):
    h1 = torch.stack(stats).t().float()
    return h1[:2].sum(0)/h1.sum(0)

def hook_stats(h, module, input, output):
    if not hasattr(h,'hook_stats'): h.hook_stats = ([],[],[])
    means,stds,hists = h.hook_stats
    means.append(output.data.mean())
    stds .append(output.data.std())
    hists.append(output.data.cpu().histc(80,0,10))