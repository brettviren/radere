#!/usr/bin/env pytest

from radere import depos

def test_contained():
    ndepos = 10000
    d = depos.random(ndepos, ((10,20),(10,20),(10,20)))
    assert d.shape[0] == ndepos
    assert d.shape[1] == 7
    d2 = depos.contained(d, ((10,15),(10,15),(10,15)))
    assert d2.shape[0] < ndepos
    assert d2.shape[1] == 7

