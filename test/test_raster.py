#!/usr/bin/env python3

import numpy
from time import time
import matplotlib.pyplot as plt
from radere.raster import Raster
from radere import units, aots
from radere.depos import Depos, load_wctnpz as load_depos

tick = 0.5*units.us
pitch = 5*units.mm
nimp = 10
pick = pitch/nimp

def driver(depos):
    rast = Raster(numpy.array([0,0,0]), numpy.array([0,1,0]),
                  tick, pitch)
    print(type(depos))
    return rast(depos)

def test_single():
    depos = Depos(numpy.array([
        #t, q  x, y,z  dL, dT
        [0,10, 0,10,0, 1*units.us, 1*units.mm]
    ]))
    got = driver(depos)

    for count, (patch, pmin, tmin) in enumerate(zip(*got)):
        print (patch.shape)
        np,nt = patch.shape

        # left, right, bottom, top
        extent = [tmin, tmin + nt*tick,
                  pmin, pmin + np*pick]

        plt.imshow(patch, extent=extent, aspect='auto');
        plt.savefig("test_raster_%02d.png" % count)


depos_file = "data/muon-depos.npz"

def with_data(device):
    t0 = time()
    depos = load_depos(depos_file, device=device)
    t1 = time()
    got = driver(depos)
    t2 = time()

    dt1 = (t1-t0)*1e6
    dt2 = (t2-t1)*1e6
    print(f'load: {dt1} us, run: {dt2} us')

    for count, (patch, pmin, tmin) in enumerate(zip(*got)):
        print (patch.shape)
        np,nt = patch.shape

        # left, right, bottom, top
        extent = [tmin, tmin + nt*tick,
                  pmin, pmin + np*pick]

        plt.imshow(patch, extent=extent, aspect='auto');
    plt.savefig("test_raster_%02d.png" % count)

def test_with_data():
    # with_data('numpy')
    #with_data('cupy')
    # with_data('cupy')
    pass

if '__main__' == __name__:
    test_raster()
