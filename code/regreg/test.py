
import numpy as np
import pylab

x = np.linspace(-2,2,101)

def f(x, eps=0.01):
    return x**2  / (2 * eps)-  eps * np.maximum(np.fabs(x/eps)-1, 0)**2 / 2.

pylab.clf(); pylab.plot(x, f(x)) ; pylab.plot(x, f(x, eps=0.1))

                 
