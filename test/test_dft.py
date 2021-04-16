#!/usr/bin/env pytest

import radere
import numpy
import torch
from radere.aots import size

def test_next_fast_len():
    for n in range(1000):
        f = radere.fft.next_fast_len(n)
        #print(f'{n} -> {f} {f-n}')

def test_dft_numpy():
    arr = numpy.array(numpy.arange(24))
    print(f'numpy: {arr}')
    farr = radere.fft.rfft(arr)
    print(f'numpy: {farr}')

def test_dft_torch():
    arr = torch.tensor(numpy.arange(24))
    print(f'torch: {arr}')
    farr = radere.fft.rfft(arr)
    print(f'torch: {farr}')

def test_halfconv():
    a1 = torch.tensor(numpy.arange(24))
    n1 = size(a1)
    a2 = torch.tensor(numpy.arange(48))
    n2 = size(a2)
    n = radere.fft.next_fast_len(n1+n2-1)
    assert(n==72)               # the -1 is undone 
    f1 = radere.fft.fft(a1,n=n)
    f2 = radere.fft.fft(a2,n=n)
    a3 = radere.fft.ifft(f1*f2)
    n3 = size(a3)
    assert(n3 == n)
    print (n1,n2,n,n3,size(f1),size(f2))
