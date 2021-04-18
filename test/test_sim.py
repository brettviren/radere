#!/usr/bin/env pytest

import os
from radere import depos, fake, field
from time import perf_counter as pc

tstdir = os.path.dirname(os.path.realpath(__file__))
srcdir = os.path.dirname(tstdir)
datdir = os.path.join(srcdir, "data")

def test_sim():
    nwires=1000
    nimp=10
    nticks=10000
    shape=(nwires*nimp, nticks)

    nblips = 100

    t1 = pc()
    bf = fake.blip_field(shape, nblips)
    t2 = pc()
    print (f'blips: {bf.shape} {t2-t1:.3f}s')
    frs = field.load_pcbro_fpwctnpz(os.path.join(datdir,"dv-2000v-h2mm5.npz"))
    t3 = pc()
    print (f'load fr: {t3-t2:.3f}s')
    for name, fra in frs.items():
        print(f'\t{name} {fra.shape}')

    # convolve ER with FR
    # setup 2D conv
