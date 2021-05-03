#!/usr/bin/env python
'''
Project (verb) depos from 3-space to response space
'''

import math
import numpy



class Depos2PT:
    def __init__(self, origin, pitch, velocity):
        '''
        Create a projection of depos from 3-space to P-T space

        Arguments are 3-vectors.

        - origin :: location of centerof "wire zero"
        - pitch :: relative vector between two wires along pitch 
        - velocity :: nominal drift velocity
        '''
        self.origin = numpy.array(origin)
        self.pitch, self.pdir = normunit(pitch)
        self.speed, self.drift = normunit(velocity)
        
    def __call__(self, depos):

        xyz = depos.block(['x','y','z']).T
        rel = xyz - self.origin
        pitches = numpy.dot(rel, self.pdir)
............come back here after making general namedtuple array        
