#!/usr/bin/env pytest
import numpy
from radere.util import Structured
from radere import aots

def bydev(device = 'numpy'):

    fields = [('a','f4'),('b','f4'),('c','f4'),('d','i4')]
    array = aots.new(numpy.arange(24).reshape(4,-1), device=device)
    asdict = dict()
    ids = aots.new(range(6), dtype='i4', device=device)
    s = Structured("Test", id=ids)
    for fd,a in zip(fields, array):
        a = aots.new(a, dtype=fd[1], device=device)
        s.stack(fd[0], a)
        asdict[fd[0]] = a
    asdict['id'] = ids
    s2 = Structured("Test", asdict)
    assert len(s) == len(s2)
    assert len(s) == 6
    for f,d in fields:
        assert f in s
        assert len(s[f]) == 6
    z = s[0]
    assert z.a == 0
    assert z.b == 6
    assert z.c == 12
    assert z.d == 18
    print (type(z), type(z.a))
    ## how to generically test type without exhaustion?
    # assert isinstance(z.a, float)
    # assert isinstance(z.b, float)
    # assert isinstance(z.c, float)
    # assert isinstance(z.d, int)

    sel = s.a<3
    print(s.a, sel)
    s2 = s[sel]
    assert len(s2) == len(s)//2

    ab1 = s.block('a','b')
    ab2 = s.block(['a','b'])
    assert (ab1 == ab2).all()
    ab1.T.shape == (6,4)


def test_structured():
    print()
    bydev("numpy")
    bydev("cupy")
    bydev("cuda")
    bydev("cpu")
          
