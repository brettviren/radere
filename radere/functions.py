#!/usr/bin/env python3
import math
import numpy
import scipy.special
import functools


## 100ns with caching.  with no caching:
# 26us with n_sigma=16
# 32us with n_sigma=128
# 91us with n_sigma=1024
@functools.cache
def gauss(n_persig, n_total):
    '''
    Return binned, positive half Gaussian distribution on n_total bins
    and where one sigma is covered by n_persig bins.

    Given a mean and sigma, the corresponding domain grid would be

    >>> x = linspace(mean, mean+sigma*n_total/n_persig, n_total)

    '''
    sqrt2 = math.sqrt(2.0)
    start = 0
    stop = n_total / (sqrt2 * n_persig)
    num = n_total + 1
    ls = numpy.linspace(start, stop, num)
    erfs = scipy.special.erf(ls)
    return 0.5*(erfs[1:] - erfs[:-1])

