#!/usr/bin/env pytest

from time import time
from radere.drift import Transport
from radere import units, aots
from radere.depos import load_wctnpz
from radere.project import Depos2PT

data_file = "data/muon-depos.npz"

def run(device):
    import test_drift
    drifted = test_drift.run(device)
    
    p = 5*units.mm
    speed = 1.6*units.mm/units.us
    project = Depos2PT(aots.new([0,0,0], device=device),
                       aots.new([0,p,0], device=device),
                       speed)
    return project(drifted)

def test_run():
    run('numpy')

def doit(device):
    
    refpln_at = 10*units.cm
    tr = Transport(refpln_at)
    depos = load_wctnpz(data_file, device=device)
    drifted = tr(depos)
    p = 5*units.mm
    speed = 1.6*units.mm/units.us
    project = Depos2PT(aots.new([0,0,0], device=device),
                       aots.new([0,p,0], device=device),
                       speed)
    ptpts = project(drifted)
    return ptpts

def test_flavors():
    doit('numpy')

def test_plot():
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    pt = doit('numpy')

    with PdfPages("test_project.pdf") as pdf:

        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(8.5,11))

        axes[0,0].hist(pt.p/units.mm)
        axes[0,0].set_xlabel('pitches [mm]')

        axes[1,0].hist(pt.dp/units.mm)
        axes[1,0].set_xlabel('dp [mm]')

        axes[0,1].hist(pt.t/units.us)
        axes[0,1].set_xlabel('times [us]')

        axes[1,1].hist(pt.dt/units.us)
        axes[1,1].set_xlabel('dt [us]')

        plt.tight_layout()
        pdf.savefig(plt.gcf())

        
