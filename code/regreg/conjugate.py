import numpy as np
from algorithms import FISTA
from smooth import linear
from quadratic import quadratic
from composite import composite

class conjugate(composite):

    def __init__(self, smooth_f, epsilon=0.01, store_argmin=True, tol=1e-8):
        self._smooth_function = smooth_f
        self._linear = linear(np.zeros(smooth_f.primal_shape))
        self._quadratic = quadratic(smooth_f.primal_shape, coef=epsilon/2.)
        self._smooth_function_linear = container(smooth_f, self._linear, self._quadratic)
        self._solver = FISTA(self._smooth_function_linear)
        self.tol = tol
        #XXX we need a better way to pass around the Lipschitz constant
        # should go in the container class
        if hasattr(smooth_f, "lipschitz"):
            self._backtrack = False
            self._smooth_function_linear.lipschitz = smooth_f.lipschitz + epsilon
        else:
            self._backtrack = True
        self._have_solved_once = False
        self.epsilon = epsilon

        self.store_argmin = store_argmin

    def smooth_objective(self, x, mode='both', check_feasibility=False):
        """
        Evaluate the conjugate function and/or its gradient

        if mode == 'both', return both function value and gradient
        if mode == 'grad', return only the gradient
        if mode == 'func', return only the function value
        """

        self._solver.debug = False

        self._linear.vector[:] = -x
        self._solver.fit(max_its=5000, tol=self.tol, backtrack=self._backtrack)
        minimizer = self._smooth_function_linear.coefs
            
        if self.store_argmin:
            self.argmin = minimizer
        if mode == 'both':
            v = self._smooth_function_linear.smooth_objective(minimizer, mode='func')
            return -v, minimizer
        elif mode == 'func':
            v = self._smooth_function_linear.smooth_objective(minimizer, mode='func')
            return -v
        elif mode == 'grad':
            return minimizer
        else:
            raise ValueError("mode incorrectly specified")
