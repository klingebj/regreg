"""
This module has a class for specifying a problem from just
a smooth function and a single penalty.

"""
from affine import identity
from atoms import atom
import numpy as np

from .composite import composite

class simple_problem(composite):
    
    def __init__(self, smooth_atom, nonsmooth_atom):
        self.smooth_atom
        self.nonsmooth_atom = nonsmooth_atom
        
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
        self.nonsmooth_atom.proximal(lipschitz, x, grad)
