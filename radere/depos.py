#!/usr/bin/env python3
'''
A single "depo" is conceptually an n-tuple:

>>> (t,q,x,y,z,long,tran,id)

The first 7 are floats, and id is an int.  Any transformation of a
depo should preserve the value of its id in the result.

Note, arrays as processed by this module are only on the 'numpy' "device".
'''

import numpy
from radere import units, util, aots

# fixme: allow cupy/torch as input without needing explicit device
def new(sarr, device='numpy'):
    '''
    Create a structured depo array with 7xN, 8xN or 7/8 element dict
    '''
    fields = ('t','q','x','y','z','long','tran')

    if isinstance(sarr, dict):
        for field in fields:
            sarr[field] = aots.new(sarr[field], dtype='f4',
                                   device=device)
        if 'id' in sarr:
            sarr['id'] = aots.new(sarr['id'], dtype='i4',
                                  device=device)

    elif isinstance(sarr, numpy.ndarray):
        d = dict()
        if not sarr.dtype.fields:
            for index, field in enumerate(fields):
                d[field] = aots.new(sarr[index], dtype='f4',
                                    device=device)
            if len(sarr) == 8:
                d["id"] = aots.new(sarr[7], dtype='i4',
                                   device=device)

        else:
            for field in fields:
                d[field] = aots.new(sarr[field], dtype='f4',
                                    device=device)
            if 'id' in d:
                d['id'] = aots.new(range(len(d['t'])), dtype='i4',
                                   device=device)
                

        sarr = d
        if 'id' not in sarr:
            sarr['id'] = aots.new(range(len(sarr['t'])), dtype='i4',
                                  device=device)

    else:
        raise TypeError(f'unknown depos array type {type(sarr)}')

    unsrt = util.Structured('Depo', **sarr)
    srt = unsrt[aots.argsort(sarr['t'])]
    return srt


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
    return new(dd, device)


def contained(depos, ranges):
    '''
    Return depos filtered to be inside the ranges.

    Ranges are a sequence of min/max for each axis
    '''
    for idim,dim in enumerate("xyz"):
        coord = depos[dim]
        hi = coord >= ranges[idim][0]
        depos = depos[hi]
        coord = depos[dim]
        lo = coord < ranges[idim][1]
        depos = depos[lo]
    return depos


def random(num, ranges, qdist=None, deltat = units.ms, device='numpy'):
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

    return new(numpy.vstack([t,q]+xyz+[dL,dT]), device=device)

