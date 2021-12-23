#!/usr/bin/env python3
'''
Simple pachinko toy reimplemented using JAX
'''
import jax.numpy as jnp
from jax.scipy.special import erf as jerf
from jax import random, jit

def rtest(key, shape):
    return random.uniform(key, shape=shape, dtype=jnp.dtype('float32'))

def random_depos(key, shape, bb=((0.0,1000.0),(-50.0,50.0)), dtype = jnp.dtype('float32')):
    '''
    Return random point depos in bounding box bb as shape
    (4,) + shape
    '''
    key1,key2 = random.split(key)

    x = random.uniform(key1, shape=shape, dtype=dtype,
                       minval=bb[0][0], maxval=bb[0][1])
    y = random.uniform(key2, shape=shape, dtype=dtype,
                       minval=bb[1][0], maxval=bb[1][1])
    q = jnp.ones(shape, dtype=dtype)
    s = jnp.zeros(shape, dtype=dtype)
    return jnp.stack([x,y,q,s])
random_depos = jit(random_depos, static_argnums=(1,2))


def drift(deposet, dt=0.01, lt=3000.0):
    '''
    Drift a set of depos, returning new ones.

    Deposet shape assumed: (4,) + rest.

    First index enumerates (x,y,q,sigma).
    '''
    # X-drift distance, output X will be at 0.0
    dx = deposet[0]
    xdrifted = jnp.zeros_like(dx)

    # Y coordinate stays same
    ydrifted = deposet[1] 

    # Fraction of depo absorbs as a function of drift distance
    relx = dx/lt
    qdrifted = deposet[2] * jnp.exp(-relx)

    # Gain Gaussian extent
    width = deposet[3]
    sdrifted = jnp.sqrt(2*dt*dx + width*width)

    # Output shape matches input
    return jnp.stack([xdrifted, ydrifted, qdrifted, sdrifted])
drift = jit(drift, static_argnums=(1,2))

@jit
def collect(drifted, binning):
    '''
    Collect a deposets into a histogram.

    The drifted is expected to be of shape: (4, ndepos)

    Result will be an array of len(binning)-1.

    The binning determines bin edges of a histogram.  I must include
    the high-side edge of the final bin.  Eg:

    >>> jnp.linspace(-52.5,52.5, 22)
    '''

    ym = drifted[1]
    q = drifted[2]
    sigma = drifted[3]

    # nbins+1 -> (nbins+1, ndepos)
    bins = jnp.broadcast_to(binning, (len(ym), len(binning))).T

    # Bring Gaussian to standard normal, (nbatches,ndepos) at each bin edge
    scaled = (bins - ym)/(sigma+0.001)
    erfs = jerf(scaled)

    # go from signed to values in [0,1] scalled by charge
    normed = 0.5*q*(1+erfs)

    # "intgegrate" over bin
    binned = normed[1:, :] - normed[:-1, :]

    # return collection of binned charge summed over all ndepos
    return jnp.sum(binned, -1)
    
def test_forward():
    key = random.PRNGKey(0)
    key,skey = random.split(key)
    # rule: a key is 'consumed' by a function call
    depos = random_depos(skey, (10,))
    # do it three different ways to make sure jit() was called right
    drifted = drift(depos)
    drifted = drift(depos, 0.01)
    drifted = drift(depos, 0.01, lt=3000.0)
    binning = jnp.linspace(-52.5,52.5, 22)
    adcs = collect(drifted, binning)
    return adcs

batched_random_depos = jit(vmap(random_depos, in_axes=(0,None)), static_argnums=(1,2))
batched_drift = jit(vmap(drift, in_axes=0))

def test_vmap_forward():
    key = random.PRNGKey(0)
    # rule: a key is 'consumed' by a function call
    nbatch=3
    skeys = random.split(key, nbatch)
    ndepos = 10
    batched_depos = batched_random_depos(skeys, (ndepos,)) # -> (nbatch, 4, ndepos)
    assert(batched_depos.shape[0] == len(skeys))
    batched_drifted = batched_drift(batched_depos,dt=0.01, lt=3000.0)
    #batched_drifted = batched_drift(batched_depos)
    # drifted = drift(depos)
    # binning = jnp.linspace(-52.5,52.5, 22)
    # adcs = collect(drifted, binning)
    

from jax import grad, vmap

