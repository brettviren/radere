#!/usr/bin/env pytest

from time import time
from radere.drift import Transport
from radere import units, aots
from radere.depos import load_wctnpz

data_file = "data/muon-depos.npz"

def doit(device, rel=1.0):
    
    refpln_at = 10*units.cm
    tr = Transport(refpln_at)
    t0 = time()
    dd = load_wctnpz(data_file, device=device)
    t1 = time()
    ddd = tr(dd)
    t2 = time()    
    dt1us = (t1-t0)*1e6
    dt2us = (t2-t1)*1e6
    if rel != 1.0:
        rel = dt2us/rel
    print(f'{device}:\tload:{dt1us:10.0f} us\tdrift:{dt2us:10.0f} us\trel: {rel:0.3f}')
    return dt2us,dd,ddd

# python -m memory_profiler test/test_drift.py
# @profile
def test_timing():
    '''
    Check time to drift
    '''
    print ('\n')
    rel,_,_ = doit("numpy")
    doit("cupy", rel)
    doit("cupy", rel)
    doit("cpu", rel)
    doit("cuda", rel)
    doit("cuda", rel)

import matplotlib.pyplot as plt

def plot_drifted(depos):

    import radere.units

    toplot=[('t','us'), ('q','e'),('x','mm'),('y','mm'),('z','mm'),
            ('long','mm'),('tran','mm')]

    fig, axes = plt.subplots(nrows=7, figsize=(8.5,11))
    for ind, tp in enumerate(toplot):
        var,unit = tp
        norm = getattr(radere.units, unit)
        val = depos[var]/norm
        axes[ind].hist(val)
        axes[ind].set_xlabel(f'{var} [{unit}]')
    plt.tight_layout()
    plt.savefig("test_drift.pdf")

def test_plots():

    rel,depo_init,depo_fini = doit("numpy")    
    assert len(depo_init) == len(depo_fini)

    plot_drifted(depo_fini)

if '__main__' == __name__:
    test_timing()
    
