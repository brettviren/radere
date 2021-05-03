#!/usr/bin/env python3
'''
Utilities
'''
from collections import namedtuple
from radere import aots
import math

def magunit(a):
    '''
    Return (magnitude, direction) or array/vector a
    '''
    amod = aots.mod(a)
    norm = math.sqrt(amod.sum(amod.dot(a,a)))
    unit = a/norm
    return norm,unit


class Structured:
    '''
    Like Numpy's structured array but on any radere device.
    '''

    # fixme: if I was smarter, this would be a type factory

    def __init__(self, name, sarray=None, **arrays):
        '''
        Create a structured array on device.

        Name is used for named tuple.

        The sarray is dict-of Numpy arrays or Numpy structured array.
        '''
        self.name = name
        self.fields = list()
        self.arrs = dict()
        self.dtypes = dict()
        if sarray is not None:
            self.intern(sarray)
        for name, array in arrays.items():
            self.stack(name, array)

    def __str__(self):
        names = ','.join(self.fields)
        length = len(self)
        return f'Structured([{names}] x {length})'

    def intern(self, sarray):
        '''
        Intern a 2D structured array or dict of 1D arrays.
        '''
        if isinstance(sarray, dict):
            for key, array in sarray.items():
                self.stack(key, array)
            return
        if hasattr(sarray, "dtype"):
            for key in sarray.dtype.fields:
                self.stack(key, sarray[key])
            return
        raise TypeError(f'unknown structured array type {type(sarray)}')
            
    def stack(self, key, arr):
        '''
        Add another array to the stack
        '''
        if self.arrs:
            if len(arr) != len(self):
                raise ValueError(f'{self.name} ragged at {key}')
        self.fields.append(key)
        self.dtypes[key] = arr.dtype
        self.arrs[key] = arr
        #print(f'{self.name} add {key} {len(arr)}')

    @property
    def Tuple(self):
        return namedtuple(self.name, self.fields)

    def __len__(self):
        if self.arrs:
            return len(self.arrs[self.fields[0]])
        return 0

    def __getattr__(self, key):
        return self.arrs[key]

    def __contains__(self, item):
        return item in self.fields

    def __getitem__(self, item):
        if isinstance(item, str):
            return self.arrs[item]
        if isinstance(item, int):
            T = self.Tuple
            return T(**{f:self.arrs[f][item] for f in self.fields})

        if isinstance(item[0], bool) or 'bool' in str(item[0].dtype):
            dat = {f:self.arrs[f][item] for f in self.fields}
            return Structured(self.name,dat)

        # assume item is sequence of indices
        return Structured(self.name,
                          {f:aots.take_along_axis(self.arrs[f], item)
                           for f in self.fields})
        
    def block(self, keys, *more):
        if isinstance(keys, str):
            keys = [keys]
        keys += more
        arrs = [self[k] for k in keys]
        amod = aots.mod(arrs[0])
        return amod.vstack(arrs)

