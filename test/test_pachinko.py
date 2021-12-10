#!/usr/bin/env python3
'''
Exercise autograd.

Define a simple model and use it to solve for input parameters as a
function of output data.

The model is an idealized depo / diffusion / drift / digitize
simpulation.  Each depo contributes to a final 1D histogram as a
Gaussian with extent that is a function of a fixed diffusion parameter
and the drift distance.

We then use the model with a given diffusion parameter value to
simulate a number of "events" each producing a 1D histogram from a
number of depos started at random drift distances.

We repeat that with a different choice for the diffusion parameter and
a different set of "events" and see if we can "train" this the first
model on the output of the second model and recover the diffusion
parameter from the second.

To keep it simple and self-contained, this is in pure torch (not
radere.aots).

A depo is a 4-tuple: (x,y,q,sigma_y).  Drifting goes from x>0 to x=0.

Caution: we assume mm for units of [distance].
'''

import numpy
import torch
import random

from torch import nn
from torch.utils.data import DataLoader, Dataset

def random_depos(count, bb=torch.tensor([(0,1000),(-50,50)])):
    '''
    Return tensor (count, 4) holding depos uniformly randomly
    distributed in given bounding box bb.
    '''
    depos = torch.zeros(count, 4)
    for axis in [0,1]:
        depos[:,axis].uniform_(bb[axis][0], bb[axis][1])
    depos[:,2] = 1.0   # for now, no q distribution
    return depos


def drift(depos, DT, lifetime):
    '''
    Drift (n,4) set of depos to x=0
    '''

    xdrift = depos[:,0]
    relx = xdrift/lifetime
    absorbprob = 1-torch.exp(-relx)

    drifted = torch.zeros_like(depos)
    drifted[:, 1] = depos[:, 1]
    drifted[:, 2] = depos[:, 2] * absorbprob
    width = depos[:, 3]
    drifted[:, 3] = torch.sqrt(2.0*DT*xdrift + width * width)
    return drifted


class Collect:
    def __init__(self, binning):
        '''
        A callable that takes drifted depos and bins their
        distribution of charge assuming Gaussian extent along
        Y-dimension.  Binning should array of bin edges, including
        high-side of last bin.
        '''
        self.binning = binning

    def __call__(self, drifted):
        ret = torch.zeros(len(self.binning)-1)
        ndepos = drifted.shape[0]
        ym = drifted[:,1]
        q = drifted[:,2]
        sigma = drifted[:,3]
        bins = self.binning.expand(ndepos, len(self.binning))
        # go to mean=0, sigma=1
        scaled = ((bins.T - ym)/sigma).T
        erfs = torch.erf(scaled)
        # go from signed to values in [0,1] scalled by charge
        # print(f'q:{q.shape}, bins:{bins.shape}, scaled:{scaled.shape}, erfs:{erfs.shape}')
        normed = 0.5*(1+erfs)
        normed = (q * normed.T).T

        binned = normed[:, 1:] - normed[:, :-1]
        # return collection of binned charge
        return torch.sum(binned, 0)


        # for one in tensor.unbind(drifted):
        #     ym = one[1]
        #     q = one[2]
        #     sigma = one[3]
        #     scaled = (self.binning - ym)/sigma
        #     erfs = torch.erf(scaled)
        #     normed = 0.5*q*(1 + erfs)
        #     binned = normed[1:] - normed[:-1]
        #     ret += binned
        # return ret

class SimpleSim(nn.Module):
    '''
    A module that drifts depos.
    '''
    def __init__(self, collector, DT=0.01, lifetime=1000):
        super(SimpleSim, self).__init__()
        self._collector = collector
        self.register_parameter(name='DT',
                                param=torch.nn.Parameter(torch.tensor(DT)))
        self.register_parameter(name='lifetime',
                                param=torch.nn.Parameter(torch.tensor(lifetime)))
    def forward(self, batch_of_depos):
        drifted = drift(batch_of_depos.reshape(-1, 4), self.DT, self.lifetime).reshape(batch_of_depos.shape)
        adcs = list()
        for depos in torch.unbind(drifted):
            adc = self._collector(depos)
            adcs.append(adc)
        return torch.vstack(adcs)


class DeposGenerator(Dataset):
    '''
    A dataset that randomly generates depos.

    The "labeler" is a function taking an (n,4) depo tensor and
    returning a "label" tensor, eg 1D ADC samples across "channels"
    '''
    def __init__(self, labeler, nevent=1000, nper=1000, 
                 bb=[(0,1000),(-50,50)]):
        
        depos = random_depos(nevent*nper, bb)
        self.depos = depos.reshape(nevent, nper, 4)
        self.labeler = labeler

    def __len__(self):
        return self.depos.shape[0]

    def __getitem__(self, idx):
        depos = self.depos[idx]
        return depos, self.labeler(depos)



def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    numpy.random.seed(worker_seed)
    random.seed(worker_seed)

def test(batch_size=10, num_workers=2, learning_rate=0.001):
    g = torch.Generator()
    g.manual_seed(0)

    real_DT = 0.01
    real_lifetime = 3000.0

    collector1 = Collect(torch.linspace(-50, 50, 21))
    collector2 = Collect(torch.linspace(-50, 50, 21))
    model = SimpleSim(collector1, DT=0.01, lifetime=3000.0)

    dg = DeposGenerator(lambda depos: collector2(drift(depos, real_DT, real_lifetime)))

    dgl = DataLoader(dg,
                    batch_size=batch_size,
                    num_workers=num_workers,
                    worker_init_fn=seed_worker,
                    generator=g,
                    )

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    # training
    size = len(dgl.dataset)
    for batch, (X, y) in enumerate(dgl):
        pred = model(X)
        loss = loss_fn(pred, y)

        # backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 10 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test2():
    depos = random_depos(1000)
    print("depos",depos)
    drifted = drift(depos, 0.01, 1000)
    print ("drifted", drifted)
    collector = Collect(torch.linspace(-50, 50, 21))
    adcs = collector(drifted)
    print ("adcs", adcs)

if '__main__' == __name__:
    test()
    #test2()
    
