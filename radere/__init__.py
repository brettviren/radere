

# Life a "module" that acts like numpy.fft or torch.fft depending on
# if its methods are called with numpy array or pytorch tensor.
from .dft import FFT
fft = FFT()

from .version import version
__version__ = version
