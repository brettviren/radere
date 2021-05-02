#!/usr/bin/env python3
'''
Uniform function interface to numpy arrays and torch tensors

We say "aot" when refering to these types generically.

'''
import numpy
import torch
import cupy

def device(aot):
    if is_torch(aot):
        return str(aot.device)
    if is_cupy(aot):
        return 'cupy'
    return 'numpy'

def is_numpy(aot):
    if isinstance(aot, str) and aot == 'numpy':
        return True
    return isinstance(aot, numpy.ndarray)

def is_torch(aot):
    if isinstance(aot, str) and aot in ('torch','cuda','cpu'):
        return True
    return isinstance(aot, torch.Tensor)

def is_cupy(aot):
    if isinstance(aot, str) and aot == 'cupy':
        return True
    # maybe a more proper type?
    return isinstance(aot, cupy._core.core.ndarray)

def mod(aotod):
    '''
    Return native module for array or tensor or device
    '''
    if is_torch(aotod):
        return torch
    if is_cupy(aotod):
        return cupy
    if is_numpy(aotod):
        return numpy
    raise ValueError(f'no module found associated with {aotod}')

def aot(dat, dev='numpy', **kwds):
    '''
    Return posibly new aot by copying dat to device.
    '''
    # Worse case, a dense (to x from) matrix must be implemented.
    # But we will only support the diagonal and to/from numpy.

    # the diagonal
    if dev == device(dat):
        return dat

    # from numpy
    if is_numpy(dat):
        if is_torch(dev):
            if dev == 'torch':
                dev = 'cpu'
            return torch.tensor(dat, device=dev, requires_grad = False)
        if is_cupy(dev):
            return cupy.array(dat, **kwds)

    # to numpy:
    if is_numpy(dev):
        if is_cupy(dat):
            return mod(dev).array(dat.get(), **kwds)

        if is_torch(dat):
            return dat.cpu().numpy()
        
    # All else are a hail mary
    return mod(dev).array(dat, **kwds)

def size(aot):
    if is_torch(aot):
        return aot.shape.numel()
    return aot.size

def shape(aot):
    '''
    Return shape of array as tensor
    '''
    if is_torch(aot):
        return tuple(aot.shape)
    return aot.shape

def extend(aot, newsize, **kwds):
    '''
    Return new array of newsize with aot set on it lower corner.
    '''
    new = zeros(newsize, device=device(aot))
    ss = tuple([slice(0, min(m)) for m in zip(aot.shape, new.shape)])
    new[ss] = aot[ss]
    return new

def zeros(shape, device='numpy'):
    '''
    Return zeros of shape on device
    '''
    if device in ('cpu', 'cuda'):
        return torch.zeros(shape, device=device, requires_grad = False)
    if device == 'cupy':
        return cupy.zeros(shape)
    return numpy.zeros(shape)

def zeros_like(aot):
    return mod(aot).zeros_like(aot)

def load_array(filename, aname, device='numpy'):
    '''
    Load array aname from filename into device.

    With device in {'numpy', 'cupy', 'cpu', 'cuda'} 
    '''
    fp = numpy.load(filename)
    arr = fp[aname]
    if device == 'numpy':
        return arr
    if device == 'cupy':
        return cupy.array(arr)
    return torch.tensor(arr, device=device, requires_grad = False)

def cast(aot, type):
    '''
    Return a new array with aot element values cast to type.
    '''
    if is_torch(aot):
        tts = {'int':'torch.IntTensor', 'float': 'torch.FloatTensor',
               'double': 'torch.DoubleTensor'}
        return aot.type(tts[type])
    return aot.astype(type)


def binomial(narr, parr):
    '''
    Return samples from binomial distributions
    '''
    if is_cupy(narr):
        narr = narr.get()
        parr = parr.get()
        return cupy.array(numpy.random.binomial(narr, parr))

    if is_numpy(narr):
        return numpy.random.binomial(narr, parr)

    from torch.distributions.binomial import Binomial
    b = Binomial(narr, parr)
    return b.sample()
