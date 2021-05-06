#!/usr/bin/env pytest
'''
Test erf/gaussian
'''

import math
import numpy
import radere.functions

def test_plots():
    import matplotlib.pyplot as plt

    n_persig = 16
    n_sig = 5
    n_total = n_sig*n_persig
    g = radere.functions.gauss(n_persig, n_total)
    norm = numpy.sum(g)
    sigma = 3.0
    mean = 1.0
    dx = sigma/n_persig
    x = numpy.linspace(mean, mean + sigma*n_sig, n_total)

    A = dx/(sigma*math.sqrt(2*math.pi))

    g2 = A * numpy.exp(-0.5*((x - mean)/sigma)**2)
    norm2 = numpy.sum(g2)
    print(len(g), len(x), len(g2))
    print(f'interf:{norm:.3f} direct:{norm2:.3f}')
    plt.plot(x,g, label="int erf")
    plt.plot(x,g2, label="direct")
    plt.legend()
    plt.savefig("test_gerf.png")

