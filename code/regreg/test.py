import gc

import numpy as np
import scipy.sparse

from .affine import power_L, normalize, selector, identity, affine_transform
from .atoms import constrained_positive_part
from .smooth import logistic_loss
from .quadratic import squared_error
from .separable import separable_problem
from .simple import simple_problem
from .identity_quadratic import identity_quadratic as iq
from .paths import lasso

class posneg(affine_transform):

    def __init__(self, X):
        n, p = X.shape
        self.X = X
        # where to store output so we don't recreate arrays 
        self._adjoint_output = np.zeros((2*p,))
        self.affine_offset = None
        self.primal_shape = (2*p,)
        self.dual_shape = (n,)

    def linear_map(self, x):
        p = self.X.shape[1]
        return  np.dot(self.X, x[:p]) - np.dot(self.X, x[p:])

    def affine_map(self, x):
        return self.linear_map(x)

    def offset_map(self, x):
        return x

    def adjoint_map(self, x):
        u = self._adjoint_output
        p = self.X.shape[1]
        u[:p] = np.dot(x, self.X)
        u[p:] = -u[:p]
        return u
