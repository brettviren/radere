#!/usr/bin/env python3
'''
A somewhat silly module providing components that produce radere
arrays from images
'''
import os
import random
from matplotlib import image

def random_filename(directory):
    '''
    Rturn a file name chosen randomly from a directory.
    '''
    for attempt in range(10):
        maybe = random.choice(os.listdir(directory))
        ext = os.path.splitext(maybe)[-1]
        if ext.lower() in (".jpeg",".jpg",".png"):
            return os.path.join(directory,maybe)


def sample(filename, shape, offset = (0,0)):
    '''
    Return a sample of image from filename of given shape and offset

    Returned array is shape + [color-depth]
    '''
    im = image.imread(filename)
    slcs = tuple([slice(o,s) for o,s in zip(offset, shape)])
    return im[slcs]

