import re
import typing
from collections import namedtuple
import torch
import math
import time
from tools import *

class CancelTrainException(Exception): pass
class CancelEpochException(Exception): pass
class CancelBatchException(Exception): pass


class Learner():
    def __init__(self, model, opt, loss_func, data):
        self.model, self.opt, self.loss_func, self.data = model, opt, loss_func, data


class Runner():
    """
    begin_fit
        begin_epoch
            begin_batch
            after_pred
            after_loss
            after_backward
            after_step
            after_batch
        begin_validate
            begin_batch
            after_pred
            after_loss
            after_backward
            after_step
            after_batch
        after_epoch
    after_fit
    """

    def __init__(self, cbs=None, cb_funcs=None):
        cbs = listify(cbs)
        for cbf in listify(cb_funcs):
            cb = cbf()
            setattr(self, cb.name, cb)
            cbs.append(cb)
        self.stop = False
        self.cbs = [TrainEvalCallback()] + cbs

    @property
    def opt(self):
        return self.learn.opt

    @property
    def model(self):
        return self.learn.model

    @property
    def loss_func(self):
        return self.learn.loss_func

    @property
    def data(self):
        return self.learn.data

    def one_batch(self, xb, yb):
        try:
            self.xb,self.yb = xb,yb
            self('begin_batch')
            self.pred = self.model(self.xb)
            self('after_pred')
            self.loss = self.loss_func(self.pred, self.yb)
            self('after_loss')
            if not self.in_train: return
            self.loss.backward()
            self('after_backward')
            self.opt.step()
            self('after_step')
            self.opt.zero_grad()
        except CancelBatchException: self('after_cancel_batch')
        finally: self('after_batch')


    def all_batches(self, dl):
        self.iters = len(dl)  # howmany batches in one epoch
        try:
            for xb,yb in dl:
                if self.stop: break
                self.one_batch(xb, yb)
        except CancelEpochException: self('after_cancel_epoch')
        finally:
            self.stop = False

    def fit(self, epochs, learn):
        self.epochs, self.learn = epochs, learn

        try:
            for cb in self.cbs: cb.set_runner(self)
            if self('begin_fit'): return
            for epoch in range(epochs):
                self.epoch = epoch
                if not self('begin_epoch'):  # through every callback's begin_epoch function
                    self.all_batches(self.data.train_dl)

                with torch.no_grad():
                    if not self('begin_validate'):
                        self.all_batches(self.data.valid_dl)
                if self('after_epoch'): break
        except CancelTrainException: self('after_cancel_fit')
        finally:
            self('after_fit')
            self.learn = None

    def __call__(self, cb_name):
        res = False
        for cb in sorted(self.cbs, key=lambda x: x._order):
            res = cb(cb_name) or res
        return res

class Callback:
    _order = 0
    def set_runner(self, run): self.run = run
    def __getattr__(self, k): return getattr(self.run, k)

    @property
    def name(self):
        name = re.sub(r'Callback$', '', self.__class__.__name__)
        return camel2snake(name or 'callback')

    def __call__(self, cb_name):
        f = getattr(self, cb_name, None)
        if f and f(): return True
        return False

class TrainEvalCallback(Callback):
    def begin_fit(self):
        self.run.n_epochs = 0.
        self.run.n_iter = 0

    def begin_epoch(self):
        self.run.n_epochs = self.epoch
        self.model.train()
        self.run.in_train = True

    def after_batch(self):
        if not self.in_train:
            return
        self.run.n_epochs += 1. / self.iters
        self.run.n_iter += 1

    def begin_validate(self):
        self.model.eval()
        self.run.in_train = False

class AvgStats():
    def __init__(self, metrics, in_train):
        self.metrics, self.in_train = listify(metrics), in_train

    def reset(self):
        self.tot_loss, self.count = 0., 0
        self.tot_mets = [0.] * len(self.metrics)

    @property
    def all_stats(self):
        return [self.tot_loss.item()] + [i.item() for i in self.tot_mets]

    @property
    def avg_stats(self):
        return [o / self.count for o in self.all_stats]

    def __repr__(self):
        if not self.count: return ""
        return f"{'train' if self.in_train else 'valid'}: {self.avg_stats}"

    def accumulate(self, run):
        bn = run.xb.shape[0]
        self.tot_loss += run.loss * bn
        self.count += bn
        for i, m in enumerate(self.metrics):
            self.tot_mets[i] += m(run.pred, run.yb) * bn

class AvgStatsCallback(Callback):
    def __init__(self, metrics, need_time=True):
        self.train_stats = AvgStats(metrics, True)
        self.valid_stats = AvgStats(metrics, False)
        self.need_time = need_time

    def begin_epoch(self):
        self.train_stats.reset()
        self.valid_stats.reset()
        self.run.epoch_ts = time.time()

    def after_loss(self):
        stats = self.train_stats if self.in_train else self.valid_stats
        with torch.no_grad():
            stats.accumulate(self.run)

    def after_epoch(self):
        self.run.epoch_ts = time.time() - self.run.epoch_ts
        time_str = f"{self.run.epoch_ts:.1f} sec" if self.need_time and self.run.epoch_ts else ''
        print(f"epoch {self.epoch+1}: {self.train_stats} {self.valid_stats} {time_str}")

class RecordCallback(Callback):
    def begin_fit(self):
        self.lrs = []
        self.losses = []

    def after_batch(self):
        self.lrs.append(self.opt.param_groups[-1]['lr'])
        self.losses.append(self.loss.detach().cpu())

class CudaCallback(Callback):
    def __init__(self,device):
        self.device=device

    def begin_fit(self):
        self.model.to(self.device)

    def begin_batch(self):
        self.run.xb = self.xb.to(self.device)
        self.run.yb = self.yb.to(self.device)

class BatchTransformXCallback(Callback):
    _order = 2
    def __init__(self, tfm):
        self.tfm = tfm

    def begin_batch(self):
        self.run.xb = self.tfm(self.run.xb)

class ParamScheduler(Callback):
    _order=1
    def __init__(self, pname, sched_funcs): self.pname,self.sched_funcs = pname,sched_funcs

    def begin_fit(self):
        if not isinstance(self.sched_funcs, (list,tuple)):
            self.sched_funcs = [self.sched_funcs] * len(self.opt.param_groups)

    def set_param(self):
        assert len(self.opt.param_groups)==len(self.sched_funcs)
        for pg,f in zip(self.opt.param_groups,self.sched_funcs):
            pg[self.pname] = f(self.n_epochs/self.epochs)

    def begin_batch(self):
        if self.in_train: self.set_param()

def annealer(f):
    def _inner(start, end): return partial(f, start, end)
    return _inner

@annealer
def sched_lin(start, end, pos): return start + pos*(end-start)

@annealer
def sched_cos(start, end, pos): return start + (1 + math.cos(math.pi*(1-pos))) * (end-start) / 2

@annealer
def sched_no(start, end, pos):  return start

@annealer
def sched_exp(start, end, pos): return start * (end/start) ** pos

def cos_1cycle_anneal(start, high, end):
    return [sched_cos(start, high), sched_cos(high, end)]

def combine_scheds(pcts, scheds):
    assert sum(pcts) == 1.
    pcts = torch.tensor([0] + callback.listify(pcts))
    assert torch.all(pcts >= 0)
    pcts = torch.cumsum(pcts, 0)
    def _inner(pos):
        idx = (pos >= pcts).nonzero().max()
        actual_pos = (pos-pcts[idx]) / (pcts[idx+1]-pcts[idx])
        return scheds[idx](actual_pos)
    return _inner

if __name__ == "__main__":
    from functools import partial
    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(torch.randn((1000, 10))), 
        batch_size=16,
        shuffle=True)
    vali_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(torch.randint(10, (1000,))), 
        batch_size=16)
    dev = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(next(iter(train_loader)))
    print(next(iter(vali_loader)))
    model = torch.nn.Sequential(
        torch.nn.Linear(10,16),
        torch.nn.ReLU(),
        torch.nn.Linear(16,10)
    )

    learn = Learner(
        model=model,
        opt=torch.optim.Adam(model.parameters(), lr=1e-2),
        loss_func=torch.nn.CrossEntropyLoss(),
        data=namedtuple('data', ['train_dl', 'valid_dl'])(train_loader, vali_loader),
    )

    cbs = [
        partial(AvgStatsCallback, lambda x: (torch.argmax(out, dim=1) == yb).float().mean()),
        RecordCallback
    ]
    runner = Runner(cb_funcs=cbs)
    runner.fit(3, learn)