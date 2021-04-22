#!/usr/bin/env python
'''
Raster depos into patches
'''

import numpy
from radere import units

class Raster:
    '''
    A component to raster depos to patches.
    '''

    def __init__(self, origin, pitch, tick=0.5*units.us, nsigma=3, nimp=10):
        '''
        Origin is a 3-D point giving center of wire 0 in the same
        coordinate system used by the depos.

        Pitch is 3-D vector from origin to next wire along pitch
        direction.

        tick is the time bin

        nsigma is the minimum truncation of the depo gaussian

        nimp is number of impact bins across wire region.
        '''
        self.origin = origin[1:]
        self.pmag = numpy.sqrt(pitch[1:].dot(pitch[1:]))
        self.pnorm = pitch[1:]/self.pmag
        self.tick = tick
        self.nsigma = nsigma
        self.nimp = nimp
        # pick = pitch tick :)
        self.pick = self.pmag * self.nimp 

    def __call__(self, depos):
        '''
        Return a list of patches, one for each depo.

        (t,q,x,y,z,long,tran)
        '''
        t = depos[:,0]
        twid = depos[:,5]*self.nsigma
        ntmin = numpy.floor((t - twid)/self.tick)
        ntmax = numpy.ceil ((t + twid)/self.tick)
        tmin = ntmin * self.tick
        tmax = ntmax * self.tick
        nt = numpy.round(ntmax - ntmin).astype('int')
        
        # convert to pitch.
        yz = depos[:,3:5]
        p = numpy.dot((yz - self.origin), self.pnorm)
        pwid = depos[:,6]*self.nsigma
        # location of min/max in units of number of pitches
        npmin = numpy.floor((p - pwid)/self.pmag)
        npmax = numpy.ceil ((p + pwid)/self.pmag)
        pmin = (npmin - 0.5) * self.pmag
        pmax = (npmax + 0.5) * self.pmag
        np = (npmax - npmin + 1).astype('int')*self.nimp

        # enter painful serial code
        patches = list()
        for ind, q in enumerate(depos[:,1]):
            lt = numpy.linspace(tmin[ind], tmax[ind], nt[ind], endpoint=False)
            lp = numpy.linspace(pmin[ind], pmax[ind], np[ind], endpoint=False)
            P,T = numpy.meshgrid(lp,lt,indexing='ij')
            dP = (P-p)/depos[:,6]
            dT = (T-t)/depos[:,5]
            patch = q * numpy.exp(-0.5*(dP*dP + dT*dT))
            patches.append(patch)
        return (patches, pmin, tmin)
        
