#!/usr/bin/env python
'''
Things related to field.
'''
import numpy

def load_pcbro_fpwctnpz(filename):
    '''
    Load "fp wct npz" file of field response.

    These files are as produced by PCBro.

    They have 2 or 3 3-D arrays of shape:

    (12,10,many)

    The 12 spans impact positions across a strip.  The 10 correspond
    to the 10 columns of FP data:

    (t,x,y,z,r0,r1,r2,r3,r4,r5)

    where the rN is the response for the path on strip N.

    Arrays are returned in radere field response form which for each
    plane is:

    (Nimp=10, Ntrode=11, Nstep=many)
    '''
    ret = dict()
    for name, arr in numpy.load(filename).items():
        # fixme: we should change pcbro to output directly what we
        # want.  Until then we do this little dance.
        half = arr.transpose([1,0,2])[4:,:,:].reshape(12*6, -1)
        other = numpy.flip(half[12:,:], axis=0)
        #other = half[12:,:]
        full = numpy.vstack((other, half))
        ret[name] = full
    return ret
