#!/usr/bin/env python3
'''
Do a full chain 
'''

import numpy
from radere.drift import Transport
from radere import units, aots
from radere.raster import Raster

depos_file = "data/muon-depos.npz"
depos_name = "depo_data_0"
resp_file = "data/reference3views-h2mm3.npz"

def doit():
    tr = Transport(20*units.cm)
    depos = aots.load_array(depos_file, depos_name)
    drifted = tr(depos)

    tick = 0.5*units.us
    pitch = 5*units.mm
    rast = Raster(numpy.array([0,0,0]), numpy.array([0,1,0]),
                  tick, pitch)
    patches = rast(drifted)


if '__main__' == __name__:
    doit()
    
