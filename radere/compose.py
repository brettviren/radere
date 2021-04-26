#!/usr/bin/env python3
'''
Compose patches.
'''

# fixme, make agnostic
import numpy

def composer(patches, offsets):
    '''
    Packages is a list of N 2D arrays of differing shape.

    Offsets is an (N,2) array giving where a patch's [0,0] index
    lands.
    '''
    shapes = numpy.array([p.shape for p in patches])
    full_shape = numpy.max(shapes + offsets, axis=0)
    out = numpy.zeros(full_shape)
    for ind in range(offsets.shape[0]):
        beg = offsets[ind]
        p = patches[ind]
        s = p.shape
        end = beg+s
        out[beg[0]:end[0], beg[1]:end[1]] += p
    return out
