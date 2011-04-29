import numpy as np
from algorithms import FISTA
from smooth import smooth_function, linear

class conjugate(object):

    def __init__(self, smooth_f):
        self._smooth_function = smooth_f
        self._linear = linear(np.zeros(smooth_f.p))
        self._smooth_function_linear = smooth_function(smooth_f, self._linear)
        self._solver = FISTA(self._smooth_function_linear)

    def smooth_eval(self, x, mode='both'):
        """
        Evaluate the conjugate function and/or its gradient

        if mode == 'both', return both function value and gradient
        if mode == 'grad', return only the gradient
        if mode == 'func', return only the function value
        """

        self._linear.vector[:] = -x
        self._solver.debug = True
        self._solver.fit(max_its=1000, tol=1.0e-10)
        coefs = self._smooth_function_linear.coefs

        if mode == 'both':
            v = self._smooth_function_linear.smooth_eval(coefs, mode='func')
            return v, coefs
        elif mode == 'func':
            v = self._smooth_function_linear.smooth_eval(coefs, mode='func')
            return v
        elif mode == 'grad':
            return coefs
        else:
            raise ValueError("mode incorrectly specified")
