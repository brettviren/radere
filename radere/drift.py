#!/usr/bin/env python3
'''
Thanks related to bulk drifting
'''
from radere import units, aots
from radere.depos import new as make_depos

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
                 fluctuate=True):
        '''
        Create a drift transport function.
        '''

        self.speed = speed
        self.lifetime = lifetime
        self.DL = DL
        self.DT = DT
        self.planex = planex
        self.fluctuate = fluctuate
        self.long_units = 1.0

    def __call__(self, depos):
        '''
        Transport depos.
        '''
        def new(var):           # want copy
            return aots.new(depos[var])

        t = new("t")
        q = new("q")
        x = depos.x
        dx = x-self.planex
        dt = dx/self.speed

        # Select forward drifting and apply diffusion to them
        fwd = dx > 0

        dtfwd = dt[fwd]
        qfwd = q[fwd]

        dL = new("long")
        dT = new("tran")

        dLfwd = dL[fwd]
        dTfwd = dT[fwd]

        amod = aots.mod(t)

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

        rows = dict(
            t=t,
            q=q,
            x=self.planex + aots.zeros_like(x),
            y=depos.y,
            z=depos.x,
            long=dL,
            tran=dT,
            id=depos.id,
        )

        ret = make_depos(rows, aots.device(depos.t))
        return ret
