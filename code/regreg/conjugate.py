import numpy as np
from algorithms import FISTA
from smooth import smooth_function, linear, l2normsq

class conjugate(object):

    def __init__(self, smooth_f, epsilon=0.01):
        self._smooth_function = smooth_f
        self._linear = linear(np.zeros(smooth_f.p))
        self._quadratic = l2normsq(smooth_f.p, l=epsilon/2.)
        self._smooth_function_linear = smooth_function(smooth_f, self._linear, self._quadratic)
        self._solver = FISTA(self._smooth_function_linear)
        #XXX we need a better way to pass around the Lipschitz constant
        # should go in the container class
        if hasattr(smooth_f, "L"):
            self._backtrack = False
            self._smooth_function_linear.L = smooth_f.L + epsilon
        else:
            self._backtrack = True
        self._have_solved_once = False
        self.epsilon = epsilon

    def smooth_eval(self, x, mode='both'):
        """
        Evaluate the conjugate function and/or its gradient

        if mode == 'both', return both function value and gradient
        if mode == 'grad', return only the gradient
        if mode == 'func', return only the function value
        """

        self._solver.debug = False

        self._linear.vector[:] = -x
        self._solver.fit(max_its=1000, tol=1.0e-08, backtrack=self._backtrack)
        minimizer = self._smooth_function_linear.coefs
            
        if mode == 'both':
            v = self._smooth_function_linear.smooth_eval(minimizer, mode='func')
            return -v, minimizer
        elif mode == 'func':
            v = self._smooth_function_linear.smooth_eval(minimizer, mode='func')
            return -v
        elif mode == 'grad':
            return minimizer
        else:
            raise ValueError("mode incorrectly specified")
