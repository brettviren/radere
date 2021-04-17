#!/usr/bin/env python3
'''
Thanks related to bulk drifting
'''
from radere import units

default_speed = 1.6*units.mm/units.us
# depends on LAr purity
default_lifetime = 8 * units.ms
# arXiv:1508.07059v2
default_DL = 7.2 * units.centimeter2 / units.second
default_DT = 12.0 * units.centimeter2 / units.second


class Transport2D:
    '''
    This component will drift groups of depos by transporting them to
    a 2D plane specifically parallel to X-axis.  If a depo is already
    below the plane then it is "backed up" in time to that plane.
    '''

    def __init__(self, planex,
                 speed=default_speed,
                 lifetime=default_lifetime,
                 DL=default_DL, DT=default_DT):
        self.speed = speed
        self.lifetime = lifetime
        self.DL = DL
        self.DT = DT
        self.planex

    def __call__(self, depos):
        '''
        Transport depos, return time ordered (N,7)
        '''

        # (t,q,x,y,z,long,tran)
        q = depos[:,1]
        x = depos[:,2]
        dx = x-self.planex
        dt = dx/self.speed

        # Select forward drifting and apply diffusion to them
        fwd = dx > 0

        dtfwd = dt[fwd]
        qfwd = q[fwd]

        dL = depos[:,5]
        dT = depos[:,6]
        dLfwd = dL[fwd]
        dTfwd = dT[fwd]

        # find change in charge due to absorbtion for fwd drift
        absorbprob = 1-numpy.exp(-dtfwd / self.lifetime)
        if self.fluctuate:
            qsign = numpy.ones_like(qfwd)
            qsign[qfwd<0] = -1.0
            dQ = qsign * binomial(numpy.abs(qfwd).astype('int'), absorbprob)
        else:
            dQ = qfwd * absorbprob
        q[fwd] = dQ

        # find broadening
        dL[fwd] = numpy.sqrt(2.0*self.DL*dtfwd + dLfwd * dLfwd)
        dT[fwd] = numpy.sqrt(2.0*self.DT*dtfwd + dTfwd * dTfwd)
        
        t = depos[:,0] + dt

        out = numpy.vstack([
            t,
            q,
            self.planex + numpy.zeros_like(x),
            depos[:,3],
            depos[:,4],
            dL,
            dT])

        return out[numpy.argsort(t)]
