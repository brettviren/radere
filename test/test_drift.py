#!/usr/bin/env pytest

from time import time
from radere.drift import Transport
from radere import units, aots


def doit(device, rel=1.0):
    
    refpln_at = 10*units.cm
    tr = Transport(refpln_at)
    t0 = time()
    dd = aots.load_array("data/haiwang-depos.npz", "depo_data_0", device)
    t1 = time()
    ddd = tr(dd)
    t2 = time()    
    dt1us = (t1-t0)*1e6
    dt2us = (t2-t1)*1e6
    if rel != 1.0:
        rel = dt2us/rel
    print(f'{device}:\tload:{dt1us:10.0f} us\tdrift:{dt2us:10.0f} us\trel: {rel:0.3f}')
    return dt2us

# python -m memory_profiler test/test_drift.py
# @profile
def test_timing():
    '''
    Check time to drift
    '''
    print ('\n')
    rel = doit("numpy")
    doit("cpu", rel)
    doit("cuda", rel)
    doit("cuda", rel)


if '__main__' == __name__:
    test_timing()
    
