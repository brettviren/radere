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
        return str(aot.device).split(":")[0]
    if is_cupy(aot):
        return 'cupy'
    if is_numpy(aot):
        return 'numpy'
    None
isdevice = device

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

torch_annoyance = {
    "bool": torch.bool,
    "uint8": torch.uint8,
    "u1": torch.uint8,
    "int8": torch.int8,
    "i1": torch.int8,
    "int16": torch.int16,
    "i2": torch.int16,
    "int32": torch.int32,
    "i4": torch.int32,
    "int64": torch.int64,
    "i8": torch.int64,
    "float16": torch.float16,
    "f2": torch.float16,
    "float32": torch.float32,
    "f4": torch.float32,
    "float64": torch.float64,
    "f8": torch.float64,
    "complex64": torch.complex64,
    "c8": torch.complex64,
    "complex128": torch.complex128,
    "c16": torch.complex128
}

def new(dat, device=None, **kwds):
    '''
    Return new aot by copying dat to device.
    '''
    if device is None:
        device = isdevice(dat) or 'numpy'

    # Numpy or array-like native is common format
    if is_numpy(dat):
        arr = dat
    elif is_torch(dat):
        arr = dat.cpu().numpy()
    elif is_cupy(dat):
        arr = dat.get()
    else:
        arr = dat               # native Python object

    if is_numpy(device):
        return numpy.array(arr, **kwds)
    if is_cupy(device):
        return cupy.array(arr, **kwds)
    if is_torch(device):
        if device == 'torch':
            device = 'cpu'
        kwds['device'] = device
        if 'dtype' in kwds:
            dtype = kwds.pop('dtype')
            if isinstance(dtype, str):
                dtype = dtype.split('.')[-1]
                kwds['dtype'] = torch_annoyance[dtype]
            elif isinstance(dtype, torch.dtype):
                kwds['dtype'] = dtype

        return torch.tensor(arr, **kwds)

    raise ValueError(f'unknown device {device}')

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

def take_along_axis(arr, indices, axis=None):
    if is_cupy(arr) or is_numpy(arr):
        return mod(arr).take_along_axis(arr, indices, axis)
    # punt for now
    old_device = device(arr)
    arr = new(arr, device='numpy')
    indices = new(indices, device='numpy')
    arr = numpy.take_along_axis(arr, indices, axis)
    return new(arr, device=old_device)

def argsort(a, axis=-1):
    return mod(a).argsort(a, axis)
    
def dot(a, b):
    if not is_torch(a):
        return mod(a).dot(a,b)
    if len(a.shape) == 1 and len(b.shape) == 1:
        return torch.dot(a,b)
    ret = list()
    for one in a:
        ret.append(torch.dot(one, b))
    return torch.tensor(ret, device=device(a))

def linspace(start, stop, num, endpoint=True, device='numpy'):
    if is_numpy(device) or is_cupy(device):
        return mod(device).linspace(start, stop, num, endpoint=endpoint)
    if endpoint:
        return torch.linspace(start,stop,num,device=device)
    return torch.linspace(start,stop,num+1,device=device)[:-1]

    
def meshgrid(a, b, indexing='xy'):
    if is_torch(a):
        mg = torch.meshgrid(a,b)
        if indexing == 'ij':
            mg = [mg[0].T, mg[1].T]
        return mg
    return mod(a).meshgrid(a, b, indexing='xy')
    
