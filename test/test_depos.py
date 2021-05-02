#!/usr/bin/env pytest

from radere import depos

def test_contained():
    ndepos = 10000
    d = depos.random(ndepos, ((10,20),(10,20),(10,20)))
    assert 'x' in d
    assert hasattr(d, 'x')
    assert len(d) == ndepos
    assert len(d[0]) == len(depos.fields)
    assert len(d['x']) == ndepos

    one = d[0]
    assert hasattr(one, 'x')

    d2 = depos.contained(d, ((10,15),(10,15),(10,15)))
    assert isinstance(d2, depos.Depos)
    assert len(d2) < ndepos


    
