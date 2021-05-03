#!/usr/bin/env pytest

from radere import depos, aots
from time import time

def ondev(device):
    ndepos = 10000
    t0 = time()
    d = depos.random(ndepos, ((10,20),(10,20),(10,20)), device=device)
    assert aots.device(d.t) == device
    t1 = time()
    assert 'x' in d
    assert 'id' in d
    assert hasattr(d, 'x')
    assert hasattr(d, 'id')
    assert len(d) == ndepos

    assert len(d['x']) == ndepos

    one = d[0]
    assert hasattr(one, 'x')

    d2 = depos.contained(d, ((10,15),(10,15),(10,15)))
    assert 'x' in d2
    assert 'id' in d2
    assert hasattr(d2, 'x')
    assert hasattr(d2, 'id')
    t2 = time()
    assert len(d2) < ndepos

    dt1 = (t1-t0)*1e3
    dt2 = (t2-t1)*1e3
    print(f'{device}:\t{len(d)} -> {len(d2)} dt1:{dt1:10.3f} ms, dt2:{dt2:10.3f} ms')
    return len(d2)

    
def test_contained():
    print()
    ondev('numpy')
    ondev('cupy')
    ondev('cupy')
    ondev('cpu')
    ondev('cuda')
    ondev('cuda')
    
