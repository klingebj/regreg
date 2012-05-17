"""
This module has a class for specifying a problem from just
a smooth function and a single penalty.

"""
import numpy as np

from .composite import composite
from .affine import identity
from .atoms import atom
from .identity_quadratic import identity_quadratic

class simple_problem(composite):
    
    def __init__(self, smooth_atom, nonsmooth_atom):
        self.smooth_atom = smooth_atom
        self.nonsmooth_atom = nonsmooth_atom
        self.coefs = self.smooth_atom.coefs = self.nonsmooth_atom.coefs

    def smooth_objective(self, x, mode='both', check_feasibility=False):
        return self.smooth_atom.smooth_objective(x, mode, check_feasibility)

    def nonsmooth_objective(self, x, check_feasibility=False):
        vn = self.nonsmooth_atom.nonsmooth_objective(x, check_feasibility=check_feasibility)
        if self.smooth_atom.quadratic is not None:
            vs = self.nonsmooth_atom.nonsmooth_objective(x, check_feasibility=check_feasibility)
            return vn + vs
        return vn

    def proximal(self, lipschitz, x, grad):
        if self.smooth_atom.quadratic is not None:
            proxq = identity_quadratic(lipschitz, x, grad)
            proxq = proxq + self.smooth_atom.quadratic
            lipschitz, x, grad = proxq.coef, -proxq.offset, proxq.linear
        return self.nonsmooth_atom.proximal(lipschitz, x, grad)

    @staticmethod
    def smooth(self, smooth_atom):
        """
        A problem with no nonsmooth part except possibly
        the quadratic of smooth_atom.

        The proximal function is (almost) a nullop.
        """
        nonsmooth = zero(smooth_atom.primal_shape)
        return simple_problem(smooth_atom, nonsmooth)
