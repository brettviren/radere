#!/usr/bin/env python3
'''
Thanks related to bulk drifting
'''
import numpy
from radere import units, aots
from radere.depos import Depos

default_speed = 1.6*units.mm/units.us
# depends on LAr purity
default_lifetime = 8 * units.ms
# arXiv:1508.07059v2
default_DL = 7.2 * units.centimeter2 / units.second
default_DT = 12.0 * units.centimeter2 / units.second

class Transport:
    '''
    This component will drift groups of depos by transporting them to
    a 2D plane specifically parallel to X-axis.  If a depo is already
    below the plane then it is "backed up" in time to that plane.
    '''

    def __init__(self, planex,
                 speed=default_speed,
                 lifetime=default_lifetime,
                 DL=default_DL, DT=default_DT,
                 fluctuate=True, temporal=True):
        '''
        Create a drift transport function.

        If temporal=True, interpret the "long" value in units of time.
        '''

        self.speed = speed
        self.lifetime = lifetime
        self.DL = DL
        self.DT = DT
        self.planex = planex
        self.fluctuate = fluctuate
        self.long_units = 1.0
        if temporal:
            self.long_units = 1.0/speed

    def __call__(self, depos, device='numpy'):
        '''
        Transport depos.

        Depos should be as radere.depos.structured or a .Depos object.

        Specifying a device will move arrays to there prior to
        operation.
        '''
        amod = aots.mod(device)

        def get(var):
            a = depos[var]
            if device == 'numpy':
                return numpy.copy(a)
            return aots.aot(a, dev=device)

        t = get("t")
        q = get("q")
        x = get("x")
        dx = x-self.planex
        dt = dx/self.speed

        # Select forward drifting and apply diffusion to them
        fwd = dx > 0

        dtfwd = dt[fwd]
        qfwd = q[fwd]

        dL = get("long")
        dT = get("tran")
        # we must have spatial units here
        dLfwd = dL[fwd]/self.long_units 
        dTfwd = dT[fwd]

        # find change in charge due to absorbtion for fwd drift
        absorbprob = 1-amod.exp(-dtfwd / self.lifetime)
        if self.fluctuate:
            qsign = amod.ones_like(qfwd)
            qsign[qfwd<0] = -1.0
            dQ = qsign * aots.binomial(aots.cast(amod.abs(qfwd),'int'),
                                       absorbprob)
        else:
            dQ = qfwd * absorbprob
        q[fwd] = dQ

        # find broadening
        dL[fwd] = amod.sqrt(2.0*self.DL*dtfwd + dLfwd * dLfwd)
        dT[fwd] = amod.sqrt(2.0*self.DT*dtfwd + dTfwd * dTfwd)
        
        t += dt

        rows = [
            t,
            q,
            self.planex + aots.zeros_like(x),
            get("y"),
            get("z"),
            dL*self.long_units, # restore units
            dT,
            get("id"),
        ]

        out = numpy.vstack([aots.aot(r, dev='numpy') for r in rows])
        return Depos(out)
