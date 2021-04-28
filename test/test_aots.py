#!/usr/bin/env pytest

from time import time
from radere import aots

data_file = "data/muon-depos.npz"

def test_load():
    t0 = time()

    dd = aots.load_array(data_file, "depo_data_0", "numpy")
    t1 = time()

    tt = aots.load_array(data_file, "depo_data_0", "cuda")
    t2 = time()

    tt = aots.load_array(data_file, "depo_data_0", "cuda")
    t3 = time()

    tt = aots.load_array(data_file, "depo_data_0", "cpu")
    t4 = time()

    print(f'numpy: {t1-t0}, cuda1: {t2-t1}, cuda2: {t3-t2}, cpu: {t4-t3}')
