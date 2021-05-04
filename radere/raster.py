#!/usr/bin/env python
'''
Raster depos into patches
'''

import numpy
from radere import units, aots

class Raster:
    '''
    A component to raster depo projections to deinterlaced patches.
    '''

    def __init__(self, impact, pitch, tick=0.5*units.us, nsigma=3, nimp=10):
        '''
        Create a rasterer.

            - impact :: which of the nimp impacts to deinterlace.

            - pitch :: the pitch distance between wires

            - tick :: time bin size.

            - nsigma :: minimum truncation of the projected gaussian.

            - nimp :: number of impact bins across wire region.
        '''
        self.impact = impact
        self.pitch = pitch
        self.tick = tick
        self.nsigma = nsigma
        self.nimp = nimp

    def __call__(self, proj):
        '''
        Perform the raster.

        Each proj results in an patch of size (np, nt).  Dimension 0
        bound by (pmin, pmax), dimension 1 by (tmin, tmax).

        Patch integral pitch and tick.
        '''
        amod = aots.mod(proj.q)

        # Time dimension
        tn_wid = self.nsigma*proj.dt/self.tick
        tn_cen = proj.t/self.tick

        # t-extent in unit of tick counts
        tn_min = amod.floor(tn_cen - tn_wid)
        tn_max = amod.ceil(tn_cen + tn_wid)
        nt = aots.new((tn_max-tn_min), dtype='i4')
        tmin = tn_min * self.tick
        tmax = tn_max * self.tick

        # Pitch dimension 
        pn_wid = self.nsigma*proj.dp/self.pitch
        pn_cen = proj.t/self.pitch

        # p-extent in unit of pitch counts
        pn_min = amod.floor(pn_cen - pn_wid)
        pn_max = amod.ceil(pn_cen + pn_wid)
        np = aots.new((pn_max-pn_min), dtype='i4')

        # We move down by 1/2 pitch and up to the impact
        padjust = self.impact/self.nimp - 0.5
        pmin = (pn_min + padjust)*self.pitch
        pmax = (pn_max + padjust)*self.pitch

        device = aots.device(proj.q)
        q = proj.q
        patches = list()
        for ind in range(len(proj)):
            lt = aots.linspace(float(tmin[ind]), float(tmax[ind]),
                               int(nt[ind]), endpoint=False,
                               device=device)
            lp = aots.linspace(float(pmin[ind]), float(pmax[ind]),
                               int(np[ind]), endpoint=False,
                               device=device)
            P,T = aots.meshgrid(lp, lt, indexing='ij')

            dT = (T-proj.t[ind])/proj.dt[ind]
            dP = (P-proj.p[ind])/proj.dp[ind]
            patch = q[ind] * amod.exp(-0.5*(dP*dP + dT*dT))
            patches.append(patch)
        return dict(patches=patches,
                    pmin=pmin, pmax=pmax, np=np,
                    tmin=tmin, tmax=tmax, nt=nt)
        
