#!/usr/bin/env python3
'''
A single "depo" is conceptually an n-tuple:

>>> (t,q,x,y,z,long,tran,id)

The first 7 are floats, and id is an int.  Any transformation of a
depo should preserve the value of its id in the result.

Note, arrays as processed by this module are only on the 'numpy' "device".
'''

import numpy
from collections import namedtuple
from radere import units

fields = ('t','q','x','y','z','long','tran','id')

# Named tuple type
Depo = namedtuple("Depo", fields)


def structure(arr):
    '''
    Create a structured depo array.
    '''
    if arr.shape[0] == 7:
        tup = [tuple(d.tolist() + [i]) for i,d in enumerate(arr.T)]
    elif arr.shape[0] == 8:
        tup = [tuple(d) for d in arr.T]
    else:
        raise ValueError(f'wrong shape array: {arr.shape}')
    dtype = [(f,'f4') for f in fields]
    dtype[-1] = ('id','i4')
    arr = numpy.array(tup, dtype=dtype)
    arr = numpy.sort(arr, order=['t','x','q'])
    return arr

class Depos:
    '''
    A namedtuple-like interface to structured array.
    '''

    def __init__(self, array):
        '''
        Create a Depos.

        The array may be raw 7xN or already structured.
        '''
        if len(array.shape) == 2:
            array = structure(array)
        self.array = array

    def __len__(self):
        return len(self.array)

    def __getattr__(self, key):
        return self.array[key]

    def __contains__(self, item):
        return item in fields

    def __getitem__(self, item):
        if isinstance(item, str):
            return self.array[item]
        if isinstance(item, int):
            return Depo(*self.array[item])
        return Depos(self.array[item])
        
    def select(self, selection):
        return Depos(self.array[selection])


def load_wctnpz(filename, ind=0):
    '''
    Load depos from WCT file.

    File is as produced by wct sio NumpyDepoSaver.

    Note, this excludes any "prior" depos.
    '''
    fp = numpy.load(filename)
    dd = fp[f'depo_data_{ind}']
    di = fp[f'depo_info_{ind}']
    if dd.shape[0] != 7 or di.shape[0] != 4:
        raise ValueError(f'not a WCT depos file: {filename}')
    # select only those depos that are not "prior" depos
    orig = di[2] == 0
    dd = dd[:,orig]
    return Depos(dd)


def contained(depos, ranges):
    '''
    Return depos filtered to be inside the ranges.

    Ranges are a sequence of min/max for each axis
    '''
    for idim,dim in enumerate("xyz"):
        coord = depos[dim]
        hi = coord >= ranges[idim][0]
        lo = coord < ranges[idim][1]
        hilo = numpy.vstack((hi,lo)).T
        inside = numpy.apply_along_axis(all, 1, numpy.vstack((hi,lo)).T)
        depos = depos.select(inside)
    return depos


def random(num, ranges, qdist=None, deltat = units.ms):
    '''
    Return some random depos inside the rectangular ranges
    '''
    if len(ranges) != 3:
        raise ValueError('ranges are a 3 pairs spanning X,Y,Z')
    if qdist is None:
        q = numpy.zeros(num) + 1000
    else:
        q = numpy.random.choice(qdist, nblips)
    t = numpy.sort(numpy.random.uniform(0, deltat, num))
    xyz = [numpy.random.uniform(r[0], r[1], num) for r in ranges]
    dL = numpy.zeros_like(t)
    dT = numpy.zeros_like(t)

    return Depos(numpy.vstack([t,q]+xyz+[dL,dT]))

