import numpy as np

from copy import copy

from algorithms import FISTA
from smooth import linear
from quadratic import quadratic
from composite import composite
from container import container

from .atoms import collapse_linear_terms
from .identity_quadratic import identity_quadratic

class conjugate(composite):

    def __init__(self, atom, quadratic=None, store_argmin=True, tol=1e-12):

        # we copy the atom because we will modify its quadratic part
        self.atom = copy(atom)

        if self.atom.quadratic is None:
            self.atom.set_quadratic(0, None, None, 0)
        
        if quadratic is not None:
            totalq = self.atom.quadratic + quadratic
            self.atom.set_quadratic(totalq.coef,
                                    totalq.offset,
                                    totalq.linear_term,
                                    totalq.constant_term)

        if self.atom.quadratic in [0, None]:
            raise ValueError('quadratic coefficient must be non-zero')

        self.solver = FISTA(self.atom)
        self.tol = tol
        #XXX we need a better way to pass around the Lipschitz constant
        # should go in the container class
        if hasattr(self.atom, "lipschitz"):
            self._backtrack = False
            # self._smooth_function_linear.lipschitz = atom.lipschitz + self.atom.quadratic.coef
        else:
            self._backtrack = True
        self._have_solved_once = False
        self.store_argmin = store_argmin

    def smooth_objective(self, x, mode='both', check_feasibility=False):
        """
        Evaluate the conjugate function and/or its gradient

        if mode == 'both', return both function value and gradient
        if mode == 'grad', return only the gradient
        if mode == 'func', return only the function value
        """

        self.solver.debug = False
        self.atom.quadratic.linear_term -= x
        self.solver.fit(max_its=5000, tol=self.tol, backtrack=self._backtrack)
        minimizer = self.atom.coefs
            
        if self.store_argmin:
            self.argmin = minimizer
        if mode == 'both':
            v = self.atom.objective(minimizer)
            return -v, minimizer
        elif mode == 'func':
            v = self.atom.objective(minimizer)
            return -v
        elif mode == 'grad':
            return minimizer
        else:
            raise ValueError("mode incorrectly specified")
        self.atom.quadratic.linear_term += x
