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
    ]).T)
    got = driver(depos)

    print (got)
    for count, patch in enumerate(got['patches']):
        # print (patch.shape)
        # np,nt = patch.shape

        # # left, right, bottom, top
        # extent = [tmin, tmin + nt*tick,
        #           pmin, pmin + np*pick]

        # plt.imshow(patch, extent=extent, aspect='auto');
        plt.imshow(patch, aspect='auto')
        plt.savefig("test_raster_%02d.png" % count)


depos_file = "data/muon-depos.npz"

def with_data(device):
    t0 = time()
    depos = load_depos(depos_file)
    t1 = time()
    got = driver(depos)
    t2 = time()

    dt1 = (t1-t0)*1e3
    dt2 = (t2-t1)*1e3
    print(f'load: {dt1} ms, run: {dt2} ms')

    tmin = min(got['tmin'])
    tmax = max(got['tmax'])
    nt = int(1 + (tmax-tmin)/tick)
    print ('tbins:', tick, nt, tmin, tmax)

    pmin = min(got['pmin'])
    pmax = max(got['pmax'])
    np = int(1 + (pmax-pmin)/pick)
    print ('pbins:', pick, np, pmin, pmax)

    charge = numpy.zeros((np,nt))
    for ind, patch in enumerate(got['patches']):

        it = int((got['tmin'][ind] - tmin)/tick)
        ip = int((got['pmin'][ind] - pmin)/pick)

        # print ('tbins:', tick, nt, tmin, tmax)
        # print ('pbins:', pick, np, pmin, pmax)
        # print(ind, (ip,it), patch.shape, charge.shape)
        charge[ip:ip+patch.shape[0],
               it:it+patch.shape[1]] += patch

    # left, right, bottom, top
    extent = [tmin, tmax, pmin, pmax]
    plt.imshow(charge, extent=extent, aspect='auto');
    plt.savefig(f"test_raster_{ind}.png")

def test_with_data():
    with_data('numpy')
    #with_data('cupy')
    # with_data('cupy')
    pass

def test_specific():
    
    rast = Raster(numpy.array([0,0,0]), numpy.array([0,pitch,0]))

    assert rast.pmag == pitch
    assert rast.pick == 0.1*pitch

    depos = Depos(numpy.array([
        #t, q  x, y,z  dL, dT
        [0,10, 0,10,0, 1*units.us, 1*units.mm]
    ]).T)

    pdata = rast(depos)
    assert len(pdata['patches']) == 1

    


if '__main__' == __name__:
    test_raster()
