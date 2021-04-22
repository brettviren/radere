#!/usr/bin/env python3

import numpy
import matplotlib.pyplot as plt
from radere.raster import Raster
from radere import units

def test_raster():
    tick = 0.5*units.us
    pitch = 5*units.mm
    nimp = 10
    pick = pitch/nimp
    rast = Raster(numpy.array([0,0,0]), numpy.array([0,1,0]),
                  tick, pitch, nimp)
    depos = numpy.array([
        #t, q  x, y,z  dL, dT
        [0,10, 0,10,0, 1*units.us, 1*units.mm]
    ])

    got = rast(depos)
    for count, (patch, pmin, tmin) in enumerate(zip(*got)):
        print (patch.shape)
        np,nt = patch.shape

        # left, right, bottom, top
        extent = [tmin, tmin + nt*tick,
                  pmin, pmin + np*pick]

        plt.imshow(patch, extent=extent, aspect='auto');
        plt.savefig("test_raster_%02d.png" % count)
