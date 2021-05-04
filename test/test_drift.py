#!/usr/bin/env pytest

import numpy
from time import time
from radere.drift import Transport
from radere import units, aots
from radere.depos import load_wctnpz

data_file = "data/muon-depos.npz"

def run(device):
    import test_depos
    depos = test_depos.run(device)
    refpln_at = 10*units.cm
    tr = Transport(refpln_at)
    return tr(depos)
    
def test_run():
    run('numpy')

def doit(device, rel=1.0):
    
    refpln_at = 10*units.cm
    tr = Transport(refpln_at)
    t0 = time()
    dd = load_wctnpz(data_file, device=device)
    assert (aots.device(dd.t) == device)
    assert (aots.device(dd.x) == device)
    assert (aots.device(dd.id) == device)
    assert len(dd) > 8
    t1 = time()
    ddd = tr(dd)
    print(f'{device}: t device {aots.device(ddd.t)}')
    assert (aots.device(ddd.t) == device)
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


def plot_drifted(dinit, dfini):
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    toplot=[('t','us'), ('q','e'),('x','mm'),('y','mm'),('z','mm'),
            ('long','us'),('tran','mm')]

    dfinis = dfini[numpy.argsort(dfini.id)]

    nbins = 100

    with PdfPages("test_drift.pdf") as pdf:

        for name,depos in [('dinit',dinit), ('dfini',dfini), ('dfinis',dfinis)]:

            fig, axes = plt.subplots(nrows=7, figsize=(8.5,11))
            axes[0].set_title(name)
            for ind, tp in enumerate(toplot):
                var,unit = tp
                norm = getattr(units, unit)
                val = depos[var]/norm
                axes[ind].hist(val, nbins)
                axes[ind].set_xlabel(f'{var} [{unit}]')
            plt.tight_layout()
            pdf.savefig(plt.gcf())

        fig, axes = plt.subplots(nrows=4, figsize=(8.5,11))        
        axes[0].plot(dfini.id, dfini.t/units.us)
        axes[0].plot(dfinis.id, dfinis.t/units.us)
        axes[0].set_xlabel('times [us] vs ids')
        axes[1].hist((dfinis.x - dinit.x)/units.mm, nbins)
        axes[1].set_xlabel('dx [mm]')

        speed_unit = units.mm/units.us
        axes[2].hist((dfinis.x-dinit.x)/(dfinis.t-dinit.t)/speed_unit)
        axes[1].set_xlabel('speed [mm/us]')

        nonzero = dinit.q!=0
        qi = dinit.q[nonzero]
        qf = dfinis.q[nonzero]
        x = dinit.x[nonzero]
        axes[3].scatter(x, (qi-qf)/qi, s=.1)
        axes[3].set_ylim(0,1)

        plt.tight_layout()
        pdf.savefig(plt.gcf())



def test_plots():

    rel,depo_init,depo_fini = doit("numpy")    
    assert len(depo_init) == len(depo_fini)
    assert numpy.all(depo_init.x != depo_fini.x)

    plot_drifted(depo_init, depo_fini)

if '__main__' == __name__:
    test_timing()
    
