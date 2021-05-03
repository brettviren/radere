#!/usr/bin/env python
'''
Project (verb) depos from 3-space to response space
'''

from radere import aots
from radere.util import magunit, Structured

class Depos2PT:
    def __init__(self, origin, pitch, speed):
        '''
        Create a projection of depos from 3-space to P-T space

        - origin :: 3-location of center of "wire zero"
        - pitch :: 3-displacement of two wires
        - speed :: nominal drift speed
        '''
        self.origin = origin
        self.pitch, self.pdir = magunit(pitch)
        self.speed = speed
        
    def __call__(self, depos):

        assert(aots.device(depos.t) == aots.device(self.origin))

        amod = aots.mod(depos.t)

        xyz = depos.block(['x','y','z']).T
        rel = xyz - self.origin
        pitches = amod.dot(rel, self.pdir)
        return Structured('Drifted',
                          p = pitches,
                          t = depos.t,
                          dp = depos.tran,
                          dt = depos.long / self.speed,
                          id = depos.id)
