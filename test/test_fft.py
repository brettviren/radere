#!/usr/bin/env pytest
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import  numpy
import math


def plot(pdf, a, tit=""):
    plt.plot(a)
    plt.title(tit)
    pdf.savefig(plt.gcf())
    plt.close();
 

def test_padding():
    x = numpy.array([math.exp(-0.0001*(x*x)) for x in range(1000)])
    X = numpy.fft.fft(x)
    x_tpadded = numpy.hstack([x, numpy.zeros_like(x)])
    X_tpadded = numpy.fft.fft(x_tpadded)

    bridge = numpy.zeros_like(X) 
    X_fpadded = numpy.hstack([X[:500], bridge, X[500:]])
    x_fpadded = 2*numpy.real(numpy.fft.ifft(X_fpadded))
    x_fdecimated = x_fpadded[1::2]
    dx = x - x_fdecimated

    with PdfPages("test_fft_padding.pdf") as pdf:
        plot(pdf,x,"x")
        plot(pdf,numpy.absolute(X),"X")        
        plot(pdf,x_tpadded,"x t-padded")
        plot(pdf,numpy.absolute(X_tpadded),"X t-padded")

        plot(pdf,numpy.absolute(X_fpadded),"X f-padded")
        plot(pdf,x_fpadded,"x f-padded")
        plot(pdf,x_fdecimated,"x f-padded t-decimated")
        plot(pdf,dx,"x - x f-padded t-decimated")
