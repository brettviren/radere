#!/usr/bin/env python3
'''
Calculate the time to perform wct sim convolution
'''
from math import log10

def fft1d(n):
    return n*log10(n)

def fft2d(n1, n2):
    return n1*fft1d(n2) + n2*fft1d(n1)



def short_conv(ns1, nt1, ns2, nt2):
    # convolve q-frame
    n = fft2d(ns1+ns2-1, nt1+nt2-1)
    # we have 10 impacts
    n *= 10
    # we then sum and inv fft2d
    n += n
    return n

def long_conv(ns1, nt1, nt3):
    # we (re) fft along combined times for each of ns1
    n = ns1*fft1d(nt1+nt3-1)
    # multiply long response onto each ns1 
    # and inverse fft
    n += n
    return n

def long_aio(ns1, nt1, ns2, nt2, nt3):
    # all in one we convolve in one double-long time
    nt = nt1+nt2+nt3-2
    ns = ns1+ns2-1
    n = fft2d(nt, ns)
    # we do it once per impact and then add
    n *= 10
    # and one more to inv
    n += n
    return n

ntick=10000
nwire=1000
njump=200
ntrode=21

s = short_conv(nwire, ntick, ntrode, njump)
l = long_conv(nwire, ntick, ntick)
sl=s+l

a = long_aio(nwire, ntick, ntrode, njump, ntick)

print(f's:{s:.1e} + l:{l:.1e} = {sl:.1e}')
print(f'aio:{a:.1e}')
r = sl/a
print(f'R:{r}')
