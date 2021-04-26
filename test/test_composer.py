#!/usr/bin/env pytest

import numpy
from time import time
from radere.compose import composer

def test_composer():
    N = 1000
    t0 = time()
    patches = list()
    offsets = list()
    for ind in range(N):
        s = numpy.random.randint(0,100, 2)
        p = numpy.random.rand(*tuple(s))
        o = numpy.random.randint(0,100, 2)
        patches.append(p)
        offsets.append(o)
    offsets = numpy.array(offsets)
    t1 = time()
    dt1 = (t1-t0)*1e3
    res = composer(patches, offsets)
    t2 = time()
    dt2 = (t2-t1)*1e3
    print(f'N={N}: prep: {dt1:.3f} ms, compose: {dt2:.3f} ms')
    print(res.shape)


if '__main__' == __name__:
    test_composer()
