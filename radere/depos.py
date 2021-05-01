#!/usr/bin/env python3
'''
A depo is a 7-tuple:

(t,q,x,y,z,long,tran)

A collection of depos are in a structured array using the above field
names.
'''

import numpy
from collections import namedtuple
from radere import units
from radere import aots

class Depos:
    '''
    A collection of depos
    '''
    # Access 1D array of these fields
    Depo = namedtuple("Depo", ('t','q','x','y','z','long','tran'))

    # The underlying as 7xN may be directly used.
    array = None

    def __init__(self, array):
        '''
        Create depos from 7xN or Nx7 array
        '''
        if array.shape[0] == 7:
            self.array = array
        elif array.shape[1] == 7:
            self.array = array.T
        else:
            raise ValueError("not a depos array")

    def __len__(self):
        return self.array.shape[1]

    def __getattr__(self, key):
        ind = self.Depo._fields.index(key)
        return self.array[ind]

    def __contains__(self, item):
        return item in self.Depo._fields

    def __getitem__(self, item):
        if isinstance(item, str):
            item = self.Depo._fields.index(item)
            return self.array[item]
        return self.Depo(*self.array[:,item])
        
    def select(self, selection):
        return Depos(self.array[:,selection])


def load_wctnpz(filename, ind=0, device='numpy'):
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
    if device != 'numpy':
        dd = aots.aot(dd, device=device)
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
    if qdist is None:
        q = numpy.zeros(num) + 1000
    else:
        q = numpy.random.choice(qdist, nblips)
    t = numpy.sort(numpy.random.uniform(0, deltat, num))
    r = [numpy.random.uniform(r[0], r[1], num) for r in ranges]
    dL = numpy.zeros_like(t)
    dT = numpy.zeros_like(t)
    # (t,q,x,y,z,long,tran)
    return Depos(numpy.vstack([t,q]+r+[dL,dT]))

