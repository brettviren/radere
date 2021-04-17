#!/usr/bin/env python
import numpy
from scipy import ndimage

from radere.aots import zeros_like

def blip_field(shape, nblips, sigma=2, distro=None):
    '''
    Return blip field.  If distro is given, draw from it for amplitude.
    '''
    if not distro:
        distro = [1]

    im = numpy.zeros(shape)
    points = im.shape * numpy.random.random((nblips, 2))
    points = points.T

    vals = numpy.random.choice(distro, nblips)
    im[(points[0]).astype('int'), (points[1]).astype('int')] = vals
    im = ndimage.gaussian_filter(im, sigma=sigma)
    return im
