#!/usr/bin/env python3

import numpy
from radere.simpleconv import ForwardHalf, SumBackwardHalf, LongConv

def test_simpleshort():
    shape = (4,6)
    kernel = numpy.array(range(12)).reshape(3,4)
    forward = ForwardHalf(kernel, shape)
    backward = SumBackwardHalf(shape)
    longc = LongConv(numpy.array(range(shape[1])), shape[1])
    q = numpy.array(range(shape[0]*shape[1])).reshape(shape)
    half1 = forward(q)
    q = numpy.array(range(shape[0]*shape[1])).reshape(shape)
    half2 = forward(q)
    final = backward([half1,half2])
    final = longc(final)
    print (final)


