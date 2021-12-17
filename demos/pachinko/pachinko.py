#!/usr/bin/env python3
'''
Exercise torch autograd and optimizer as a self-contained
"differentiable simulation" of very toy LArTPC detector.

The toy model is that of an idealized depo / diffusion / drift /
digitize simulation.

Each depo with charge q contributes to a final 1D histogram as
determined by its initial location (x, y).  This histogram most
closely resembles a single ADC sample from each channel which is
integrated over the entire readout/drift duration.

The depos drift to x=0 and in the process develop a finite Gaussian
extent with:

    - mean transverse position y

    - mean amplitude (electron charge) after elecron reabsorption from
      a function of x and parameter "lifetime" giving the mean
      electron liffetime of the LAr (here, in units of distance).

    - standard deviation (sigma) from a function of x and the
      transferse diffusion parameter "DT".

The toy works by defining two instances of the model, one serving as
"reality" and the other serving as the model to optimize to match
reality.  A dataset generates initial depos and transforms them
according to "reality".

Caveats and problems:

    - We implicitly assume mm for units of [distance].

    - Ignoring the fact we actually collect ADC waveforms over readout
      duration instead of a single sample.

    - No fluctuation of total electrons per depo nor across the
      Gaussian extent.

    - The dataset represents a "rectangular" array of depos.  Each
      "event" has the same number of depos.  Closer to reality this
      number will vary broadly.

Notes:

    - Must assure no_grad for dataset or else with num_workers arg to
      dataload one gets crypitic error.  See: https://redd.it/h7r6dt

    - On i7-4770K CPU: 28s (load time  6.0s, 1epoch 21.7s)
    - On i7-9750H CPU: 23s (load time  4.8s, 1epoch 17.5s),
    - On GTX 1650 GPU: 55s (load time 11.9s, 1epoch 42.9s).
'''

import time
import numpy
import torch
import random

from torch import nn
from torch.utils.data import DataLoader, Dataset


class Drifter(nn.Module):
    '''
    Drift depos, producing 1-to-1 with output gaining Gaussian extent.
    '''

    range_dt = torch.tensor([0.0,1.0])
    range_lt = torch.tensor([0.0,10000.0])

    # logit:        [0,1] -> [-inf,inf]
    # sigmoid: [-inf,inf] -> [0, 1]

    def p2dt(self, p):
        return self.range_dt[0] + (self.range_dt[1]-self.range_dt[0])*p
    def p2lt(self, p):
        return self.range_lt[0] + (self.range_lt[1]-self.range_lt[0])*p

    def dt2p(self, dt):
        return (dt - self.range_dt[0]) / (self.range_dt[1] - self.range_dt[0])
    def lt2p(self, lt):
        return (lt - self.range_lt[0]) / (self.range_lt[1] - self.range_lt[0])

    @property
    def DT(self):
        return self.p2dt(torch.sigmoid(self.param_dt))
    @property
    def lifetime(self):
        return self.p2lt(torch.sigmoid(self.param_lt))
        
    def __init__(self, DT=0.01, lifetime=3000.0):
        '''
        Create a drifter with a nominal parameter values.
        '''
        super(Drifter, self).__init__()
        initial_dt = torch.logit(self.dt2p(torch.tensor(DT))).item()
        initial_lt = torch.logit(self.lt2p(torch.tensor(lifetime))).item()
        # working parameters are [-inf,inf]
        self.param_dt = torch.nn.Parameter(torch.tensor(initial_dt))
        self.param_lt = torch.nn.Parameter(torch.tensor(initial_lt))
        
    def forward(self, batches_of_depos):
        #print('Drifter:',batches_of_depos.shape)

        # (nperbatch, ndepos, 4)
        # each depo is handled independently, w/out regards to batching
        flat = batches_of_depos.reshape(-1, 4).to(device=self.param_dt.device)

        # output
        drifted = torch.zeros_like(flat, device=self.param_dt.device)

        # X coordiates, output are all at 0.0
        xdrift = flat[:,0]

        # Y coordinate stays same
        drifted[:, 1] = flat[:, 1] 

        # Fraction of depo absorbs as a function of drift distance
        relx = xdrift/self.lifetime
        drifted[:, 2] = flat[:, 2] * torch.exp(-relx)

        # Gain Gaussian extent
        width = flat[:, 3]
        drifted[:, 3] = torch.sqrt(2.0*self.DT*xdrift + width * width)

        # Output shape matches input
        return drifted.reshape(batches_of_depos.shape)



class Collector(nn.Module):
    '''
    Each drifted depo has width and contributes electron charge to
    some bins of an output histogram.
    '''

    def __init__(self, binning):
        super(Collector, self).__init__()
        self.binning = binning

    def forward(self, drifted_depos):
        #print('Collector:',drifted_depos.shape, len(drifted_depos.shape))
        # may bet (nbatch, ndepos, 4) or (ndepos, 4)

        if len(drifted_depos.shape) == 2:
            return self.forward_one(drifted_depos)

        ret = torch.stack([self.forward_one(one) for one in drifted_depos.unbind()])
        #print('Collector returns:',ret.shape)
        return ret;

    def forward_one(self, drifted_depos):
        assert(len(drifted_depos.shape) == 2)
        assert(drifted_depos.shape[1] == 4)
        # (ndepos, 4)

        ndepos = drifted_depos.shape[0]

        # every one of ndepos contributes to its batch histogram
        ret = torch.zeros(len(self.binning)-1, device=drifted_depos.device)

        ym = drifted_depos[:,1]
        q = drifted_depos[:,2]
        sigma = drifted_depos[:,3]

        # (nbins, ndepos)
        bins  = self.binning.expand(ndepos, -1).T

        # Bring Gaussian to standard normal
        scaled = (bins - ym)/(sigma+0.001)
        erfs = torch.erf(scaled)

        # go from signed to values in [0,1] scalled by charge
        # print(f'q:{q.shape}, bins:{bins.shape}, scaled:{scaled.shape}, erfs:{erfs.shape}')
        normed = 0.5*q*(1+erfs)

        binned = normed[1:, :] - normed[:-1, :]
        # return collection of binned charge summed over all ndepos
        got = torch.sum(binned, -1)
        return got


def random_depos(nbatches, nperevent, bb=[(0,1000),(-50,50)], device='cpu'):
    '''
    Return tensor of shape (nbatches, nperevent, 4) holding depos
    uniformly randomly distributed in given bounding box bb.

    The depos are NOT made on the device.
    '''
    with torch.no_grad():
        depos = torch.zeros(nbatches, nperevent, 4, device=device)
        for axis in [0,1]:
            depos[:,:,axis].uniform_(bb[axis][0], bb[axis][1])
        depos[:,:,2] = 1.0   # for now, no q distribution
        return depos


class Deposet(Dataset):
    '''
    A dataset of depos of shape (nbatches, nperevent, 4).

    The "transform" is a callable to convert generated depos to
    dependent values: y = model(X).

    Return X, y
    '''
    def __init__(self, depos, transform):
        self.depos = depos
        with torch.no_grad():
            self.labels = transform(depos)

    def __len__(self):
        return self.depos.shape[0]

    def __getitem__(self, idx):
        if idx >= len(self):
            print("Deposet overrun!")
            raise StopIteration
        try:
            ret = self.depos[idx], self.labels[idx]
        except RuntimeError:
            print(f'idx={idx}')
            raise
        return ret


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    numpy.random.seed(worker_seed)
    random.seed(worker_seed)

def loop_train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    drifter = list(model.children())[0]
    for batch, (X, y) in enumerate(dataloader):
        pred = model(X)
        loss = loss_fn(pred, y)

        # backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 10 == 0:
            loss, current = loss.item(), batch * len(X)
            np = {np[0]:np[1] for np in model.named_parameters()}
            DT = drifter.DT.item()
            lt = drifter.lifetime.item()
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}] DT:{DT} lt:{lt}")

def loop_test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss = 0

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            #correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    #correct /= size
    print(f"Test Error: \n Accuracy: Avg loss: {test_loss:>8f} \n")    


def make_model(DT=0.01, lifetime=3000.0, pitch=5.0, nwires=21, device='cpu'):
    drifter = Drifter(DT=DT, lifetime=lifetime)
    span = nwires * pitch
    half = 0.5*span
    bins = torch.linspace(-half, half, nwires+1, device=device)
    collector = Collector(binning=bins)
    return nn.Sequential(drifter, collector)
                


def test_train(epochs = 1,
               batch_size=64, learning_rate=0.0001,  num_workers=0,
               nevent=100000, nper=10,
               device='cpu'
               ):
    t0 = time.time()
    rng = torch.Generator()
    rng.manual_seed(0)

    start_DT = 0.02
    start_lt = 2000.0

    nwires = 21
    with torch.no_grad():
        reality = make_model(0.01, 3000.0, nwires=nwires, device=device)
        reality.to(device=device)
    model = make_model(start_DT, start_lt, nwires=nwires, device=device)
    model.to(device=device)

    # get drifter so we can print its parameters
    drifter = list(model.children())[0]

    depos = random_depos(nevent, nper, device=device)
    dg = Deposet(depos, reality)

    dgl = DataLoader(dg,
                     batch_size=batch_size,
                     num_workers=num_workers, # see Notes at top
                     worker_init_fn=seed_worker,
                     generator=rng,
                     )


    loss_fn = nn.MSELoss()
    for par in model.named_parameters():
        print("parameter:", par)

    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    t1 = time.time()
    print(f'load time: {t1-t0}')

    tloop1 = time.time()
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}\n-------------------------------")
        loop_train(dgl, model, loss_fn, optimizer)
        #loop_test(dgl, model, loss_fn)
        tloop2 = time.time()
        print(f'time: {tloop2-tloop1}')
        tloop1 = tloop2

    for par in model.named_parameters():
        print("parameter:", par, par[1].grad)
    print("Done!")        


def test_gen():
    '''
    Make the dataset + dataloader and get one element out.
    '''
    rng = torch.Generator()
    rng.manual_seed(0)

    nwires = 21
    with torch.no_grad():
        reality = make_model(nwires=nwires)

    nevent=10000
    nper=100
    depos = random_depos(nevent, nper)
    dg = Deposet(depos, reality)
    print(dg[0][0].shape, dg[0][1].shape)

    batch_size = 2
    dgl = DataLoader(dg,
                     batch_size=batch_size,
                     num_workers=1, # see Notes at top
                     worker_init_fn=seed_worker,
                     generator=rng,
                     )

    two = next(iter(dgl))
    print("from dataloader:",len(two))
    assert(len(two) == 2)
    print("from dataloader:",two[0].shape, two[1].shape)
    Xs,ys = two
    assert(len(Xs.shape)==3)
    assert(len(ys.shape)==2)
    Xs_shape = (batch_size, nper, 4)
    assert(Xs.shape == Xs_shape)
    ys_shape = (batch_size, nwires)
    assert(ys.shape == ys_shape)

if '__main__' == __name__:
    #test_gen()
    nper = 10
    batch_size = 100
    nevent = batch_size * 1000
    test_train(epochs=1, batch_size=batch_size, learning_rate=0.1,
               num_workers=0,
               nevent=nevent, nper=nper,
               #device='cuda'
               device='cpu'
               )

