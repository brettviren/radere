#!/usr/bin/env python3
'''
A depo is a 7-tuple:

(t,q,x,y,z,long,tran)

When multiple depos are involved they are provided as 2-D array of
shape (N,7).  
'''

import numpy
from radere import units

def load_wctnpz(filename, ind=0):
    '''
    Return depos array from WCT filename.

    File is as produced by wct sio NumpyDepoSaver.

    Note, this excludes any "prior" depos.
    '''
    fp = numpy.load(filename)
    dd = fp[f'depo_data_{ind}'].T
    di = fp[f'depo_info_{ind}'].T
    orig = di[:2] == 0
    return dd[orig,:]


def contained(depos, ranges):
    '''
    Return depos filtered to be inside the ranges.

    Ranges are a sequence of min/max for each axis
    '''
    for dim in range(3):
        coord = depos[:,dim+2]
        hi = coord >= ranges[dim][0]
        lo = coord < ranges[dim][1]
        hilo = numpy.vstack((hi,lo)).T
        inside = numpy.apply_along_axis(all, 1, numpy.vstack((hi,lo)).T)
        depos = depos[inside, :]
    return depos


def random(num, ranges, qdist=None, deltat = units.ms):
    '''
    Return some random depos inside the rectangular ranges
    '''
    if qdist is None:
        q = numpy.zeros(num) + 1000
    else:
        q = numpy.random.choice(qdist, nblips)
    t = numpy.sort(numpy.random.uniform(0, deltat, num))
    r = [numpy.random.uniform(r[0], r[1], num) for r in ranges]
    dL = numpy.zeros_like(t)
    dT = numpy.zeros_like(t)
    # (t,q,x,y,z,long,tran)
    return numpy.vstack([t,q]+r+[dL,dT]).T

