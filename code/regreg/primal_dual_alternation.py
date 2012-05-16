import numpy as np
from scipy import sparse
from algorithms import FISTA
from composite import (composite, nonsmooth as nonsmooth_composite,
                       smooth as smooth_composite)
from .affine import (vstack as afvstack, identity as afidentity, power_L,
                    selector as afselector,
                    scalar_multiply, adjoint)
from separable import separable
from smooth import smooth_atom, affine_smooth
from atoms import affine_atom as nonsmooth_affine_atom
from cones import zero_constraint, zero as zero_nonsmooth, affine_cone

class problem_spec(composite):
    """
    A class for specifying a problem of the form

    .. math::

       \minimize_x f(x) + \sum_i g_i(D_ix)

    which will be solved by a dual problem

    .. math::

       \maximize_{u_i} -f^*(-\sum_i D^Tu_i) + \sum_i g_i^*(u_i)


    """
    def __init__(self, f, *g):
        self.f = f
        self.f_conjugate = f.conjugate
        if not isinstance(self.f_conjugate, smooth_composite):
            raise ValueError('the conjugate of f should be a smooth_composite')

        if len(g) == 0:
            g = [zero_nonsmooth(f.primal_shape)]

        duals = [gg.dual for gg in g]
        self.atransforms = [d[0] for d in duals]
        self.atoms = [d[1] for d in duals]

        transform, _ = self.dual
        self.affine_fc = affine_smooth(self.f_conjugate, scalar_multiply(adjoint(transform), -1))
        self.coefs = np.zeros(self.affine_fc.primal_shape)

    @property
    def dual(self):
        if not hasattr(self, "_dual"):
            if len(self.atransforms) > 1:            
                transform = afvstack(self.atransforms)
                transform = afvstack(transforms)
                nonsm = separable(transform.dual_shape, self.atoms,
                                  transform.dual_slices)
                self._dual = transform, nonsm
            else:
                self._dual = (self.atransforms[0], self.atoms[0])
        return self._dual

    def smooth_objective(self, x, mode='both', check_feasibility=False):
        """
        The smooth_objective DOES NOT INCLUDE the identity
        quadratic of all the smooth atoms.
        """
        v = self.affine_fc.smooth_objective(x, mode=mode, check_feasibility=check_feasibility)
        # retain a reference
        if mode in ['both', 'grad']:
            self.primal = self.affine_fc.grad
        return v

    def nonsmooth_objective(self, x, check_feasibility=False):
        out = 0.
        for atom in self.atoms:
            out += atom.nonsmooth_objective(x,
                                            check_feasibility=check_feasibility)
        if self.quadratic is None:
            return out
        else:
            return out + self.quadratic.objective(x, 'func')

    default_solver = FISTA
    def proximal(self, lipschitz, x, grad, prox_control=None):
        """
        The proximal function for the primal problem
        """
        transform, separable_atom = self.dual
        
        if isinstance(transform, afselector):
            z = y.copy()
            z[transform.index_obj] = separable_atom.proximal(lipschitz,
                                                             x[transform.index_obj],
                                                             grad[transform.index_obj])
            return z
        else:
            return separable_atom.proximal(lipschitz, x, grad)

    def _dual_smooth_objective(self,v,mode='both', check_feasibility=False):

        """
        The smooth component and/or gradient of the dual objective        
        """
        
        # residual is the residual from the fit
        transform, _ = self.dual
        residual = self._dual_response - transform.adjoint_map(v)

        if mode == 'func':
            return (residual**2).sum() / 2.
        elif mode == 'both' or mode == 'grad':
            g = -transform.linear_map(residual)
            if mode == 'grad':
                return g
            if mode == 'both':
                return (residual**2).sum() / 2., g
        else:
            raise ValueError("Mode not specified correctly")

