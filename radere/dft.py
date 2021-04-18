#!/usr/bin/env python3
'''
radere support for discrete Fourier transforms

Use it as

>>> import radere >>> frequency = radere.fft(interval)

See scipy.fft and torch.fft for the majority of the API supported.
In reading this module, the term "aot" means array-or-tensor.
'''


import numpy
import scipy.fft
import torch

from radere import aots

def next_fast_len(target, real=False):
    'Return a fast FFT size not smaller than n'
    return scipy.fft.next_fast_len(target, real)


def fast_len2(aot1, ato2):
    'Return fast size suitable for FFT-based convolution'
    return fft.next_fast_len(aots.size(aot1)+aots.size(aot2)-1)


def call(funcname, aot, *args, **kwds):
    '''
    Call meth from numpy, scipy or torch based on if aot is array or tensor.
    '''
    if isinstance(aot, torch.Tensor):
        func = getattr(torch.fft, funcname)
        return func(aot, *args, **kwds)

    func = getattr(scipy.fft, funcname)
    return func(aot, *args, **kwds)


class FFT:
    '''
    Common interface to numpy.fft or torch.fft
    '''
    def __init__(self):
        pass

    def __getattr__(self, name):
        g = globals()
        if name in g:
            return g[name]

        def callit(aot, *args, **kwds):
            return call(name, aot, *args, **kwds)
        return callit

