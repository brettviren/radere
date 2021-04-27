#!/usr/bin/env python
'''
The core convolution
'''

from radere.aots import extend, zeros_like, mod
from radere import fft

class ForwardHalf:
    '''
    The "simple" 2-D half convolution.

    It is simple in that it does not accomodate arbitrary length
    input.  It is half because it produces output in Fourier dual
    space of frequency.
    '''
    def __init__(self, kernel, shape):
        '''
        Create a simple half convolution.

        The kernel is 2-D with shape (ntrode, njump).

        The shape gives desired output size.
        '''
        self.shape = tuple([a+b-1 for a,b in zip(shape, kernel.shape)])
        fr = extend(kernel, self.shape)
        fr = fft.fft(fr, axis=0) # per tick, along wire
        fr = fft.fft(fr, axis=1) # per wire, along tick
        self.kernel = fr

    def __call__(self, qframe):
        '''
        Perform 2-D half convolution of kernel with qframe.

        The returned array holds the convolution of qframe and kernel
        left in frequency space.
        '''
        q = extend(qframe, self.shape)
        q = fft.fft(q, axis=0)
        q = fft.fft(q, axis=1)
        return q * self.kernel

class SumBackwardHalf:
    '''
    Take N from SimpleHalf, sum and do inverse 2-D FFT

    This makes no attempt to handle overlap-add.
    '''
    def __init__(self, shape):
        self.shape = shape
        pass
    def __call__(self, halves):
        q = zeros_like(halves[0])
        for one in halves:
            q += one
        q = fft.ifft(q, axis=1)
        q = fft.ifft(q, axis=0)
        return q[:self.shape[0], :self.shape[1]].real
    
class LongConv:
    '''
    Convolve a slow reponse in a per-wire basis.
    '''
    def __init__(self, kernel, tsize):
        self.tsize = tsize
        self.fsize = tsize+kernel.size-1
        fr = extend(kernel, (self.fsize,))
        fr = fft.fft(fr)
        self.kernel = fr
    def __call__(self, q):
        qout = zeros_like(q)
        for iwire in range(q.shape[0]):
            a = extend(q[iwire], (self.fsize,))
            a = fft.fft(a)
            qout[iwire] = fft.ifft(a*self.kernel)[:self.tsize].real
        return qout
    
