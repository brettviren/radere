#!/usr/bin/env python
'''
Convolve images
'''

import os
import numpy
from radere.simpleconv import ForwardHalf, SumBackwardHalf, LongConv
from radere.images import random_filename, image, sample

# relies on a collection of images.
image_dir = "/home/bv/Desktop/bkg"

def test_random():
    bigfn = random_filename(image_dir)
    print("frame image:",bigfn)
    lilfn = random_filename(image_dir)
    print("kernel image:",lilfn)

    n1 = os.path.splitext(os.path.basename(bigfn))[0]
    n2 = os.path.splitext(os.path.basename(lilfn))[0]
    outname = f"test-silly-{n1}-{n2}.jpg"

    big = image.imread(bigfn)
    lil = sample(lilfn, (200, 21))

    layers = list(range(big.shape[-1]))
    shape = big.shape[:-1]

    ffs = [ForwardHalf(lil[:,:,layer], shape) for layer in layers]
    bb = SumBackwardHalf(shape)

    frames = list()
    for layer in range(3):
        ff = ffs[layer](big[:,:,layer])
        frames.append(ff)
    print([f.shape for f in frames])
    newimage = bb(frames)
    #newimage = numpy.dstack(frames)
    print(newimage.shape)

    vmax = numpy.max(newimage)
    vmin = numpy.min(newimage)
    normed = 255.999 * (newimage - vmin) / (vmax - vmin)
    eight = normed.astype(numpy.uint8)
    print("saving:", outname, eight.dtype, eight.shape)
    image.imsave(outname, eight, cmap='gray')

if '__main__' == __name__:
    test_random()
    
