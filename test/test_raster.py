#!/usr/bin/env python3

import numpy
from time import time
import matplotlib.pyplot as plt

from radere import units, aots
from radere.depos import load_wctnpz as load_depos
from radere.drift import Transport
from radere.project import Depos2PT
from radere.raster import Raster

tick = 0.5*units.us
pitch = 5*units.mm
nimp = 10
pick = pitch/nimp
impact = 1
speed = 1.6*units.mm/units.us

def run(device):
    import test_project
    proj = test_project.run(device)
    pdat = Raster(impact, pitch)
    
def test_run():
    run('numpy')


depos_file = "data/muon-depos.npz"

def with_data(device):
    t0 = time()
    depos = load_depos(depos_file, device=device)

    t1 = time()
    refpln_at = 10*units.cm
    drift = Transport(refpln_at)
    drifted = drift(depos)

    t2 = time()
    p = pitch
    project = Depos2PT(aots.new([0,0,0], device=device),
                       aots.new([0,p,0], device=device),
                       speed)
    proj = project(drifted)


    t3 = time()
    raster = Raster(impact, pitch)
    rast = raster(proj)

    t4 = time()

    dt1 = (t1-t0)*1e3
    dt2 = (t2-t1)*1e3
    dt3 = (t3-t2)*1e3
    dt4 = (t4-t3)*1e3
    print(f'{device}:\t{dt1:10.3f} ms, {dt2:10.3f} ms, {dt3:10.3f} ms, {dt4:10.3f} ms')


def test_with_data():
    print()
    with_data('numpy')
    with_data('cpu')
    with_data('cuda')
    with_data('cuda')
    with_data('cupy')
    with_data('cupy')



if '__main__' == __name__:
    test_raster()
