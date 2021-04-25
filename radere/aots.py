#!/usr/bin/env python3
'''
Uniform function interface to numpy arrays and torch tensors

We say "aot" when refering to these types generically.

'''
import numpy
import torch


def mod(aot):
    if isinstance(aot, torch.Tensor):
        return torch
    return numpy

def size(aot):
    if isinstance(aot, torch.Tensor):
        return aot.shape.numel()
    return aot.size()

def shape(aot):
    if isinstance(aot, torch.Tensor):
        return tuple(aot.shape)
    return aot.shape

def pad(aot, newsize):
    new = mod(aot).zeros(newsize)
    ss = tuple([slice(0, min(m)) for m in zip(aot.shape, new.shape)])
    new[ss] = aot[ss]
    return new

def zeros_like(aot):
    return mod(aot).zeros_like(aot)

def load_array(filename, aname, device='numpy'):
    fp = numpy.load(filename)
    arr = fp[aname]
    if device == 'numpy':
        return arr
    return torch.tensor(arr, device=device)

def cast(aot, type):
    if isinstance(aot, numpy.ndarray):
        return aot.astype(type)
    tts = {'int':'torch.IntTensor', 'float': 'torch.FloatTensor',
           'double': 'torch.DoubleTensor'}
    return aot.type(tts[type])

def binomial(narr, parr):
    if isinstance(narr, numpy.ndarray):
        return numpy.random.binomial(narr, parr)
    from torch.distributions.binomial import Binomial
    b = Binomial(narr, parr)
    return b.sample()
