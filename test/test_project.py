#!/usr/bin/env pytest

from time import time
from radere.drift import Transport
from radere import units, aots
from radere.depos import load_wctnpz
from radere.project import Depos2PT

data_file = "data/muon-depos.npz"

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
    print (ptpts)

def test_chain():
    doit('numpy')
