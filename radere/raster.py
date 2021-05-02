#!/usr/bin/env python
'''
Raster depos into patches
'''

import numpy
from radere import units, aots

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
        self.origin = numpy.array(origin[1:])
        pitch = numpy.array(pitch)
        self.pmag = numpy.sqrt(pitch[1:].dot(pitch[1:]))
        self.pnorm = pitch[1:]/self.pmag
        self.tick = tick
        self.nsigma = nsigma
        self.nimp = nimp
        # pick = pitch tick :)
        self.pick = self.pmag / self.nimp 

    def __call__(self, depos, device='numpy'):
        '''
        Return a list of patches, one for each depo.
        '''
        amod = aots.mod(device)
        def todev(a):
            return aots.aot(a, dev=device)
        def get(var):
            a = depos[var]
            if device == 'numpy':
                return numpy.copy(a)
            return todev(a)

        t = get('t')
        dlong = get('long')
        dtran = get('tran')

        twid = dlong*self.nsigma
        ntmin = amod.floor((t - twid)/self.tick)-1
        ntmax = amod.ceil ((t + twid)/self.tick)+1
        tmin = ntmin * self.tick
        tmax = ntmax * self.tick
        nt = amod.round(ntmax - ntmin).astype('int')
        
        # convert to pitch.
        yz = todev(depos.block(['y','z']).T) # (N,2)
        p = amod.dot(yz - todev(self.origin), todev(self.pnorm))
        pwid = dtran*self.nsigma
        # location of min/max in units of number of pitches
        npmin = amod.floor((p - pwid)/self.pmag)
        npmax = amod.ceil ((p + pwid)/self.pmag)
        # the actual boundary extended to edge of wire pitch region
        pmin = (npmin - 0.5) * self.pmag
        pmax = (npmax + 0.5) * self.pmag
        # number of pick bins, add 1 for the +/- 0.5 above
        np = ((npmax - npmin + 1) * self.nimp).astype('int')

        q = depos['q']
        patches = list()
        for ind in range(len(depos)):
            lt = amod.linspace(float(tmin[ind]), float(tmax[ind]),
                               int(nt[ind]), endpoint=False)
            lp = amod.linspace(float(pmin[ind]), float(pmax[ind]),
                               int(np[ind]), endpoint=False)
            P,T = amod.meshgrid(lp,lt,indexing='ij')

            dT = (T-t[ind])/dlong[ind]
            dP = (P-p[ind])/dtran[ind]
            patch = q[ind] * amod.exp(-0.5*(dP*dP + dT*dT))
            patches.append(patch)
        return dict(patches=patches,
                    pmin=pmin, pmax=pmax, np=np,
                    tmin=tmin, tmax=tmax, nt=nt)
        
