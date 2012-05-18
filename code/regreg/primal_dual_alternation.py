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

class dual_problem(composite):
    """
    A class for specifying a problem of the form

    .. math::

       \minimize_x f(x) + \sum_i g_i(D_ix)

    which will be solved by a dual problem

    .. math::

       \maximize_{u_i} -f^*(-\sum_i D^Tu_i) + \sum_i g_i^*(u_i)

    while the primal variable is stored in the computation
    of the gradient of :math:`f^*`.

    """

    def __init__(self, f, transform, separable_atom):
        self.f = f
        self.f_conjugate = f.conjugate
        if not isinstance(self.f_conjugate, smooth_composite):
            raise ValueError('the conjugate of f should be a smooth_composite')

        self.transform = transform
        self.separable_atom = separable_atom

        # the dual problem has f^*(-D^Tu) as objective
        self.affine_fc = affine_smooth(self.f_conjugate, scalar_multiply(adjoint(self.transform), -1))
        self.coefs = np.zeros(self.affine_fc.primal_shape)

    @staticmethod
    def fromseq(f, *g):
        transform, separable_atom = stacked_dual(f.primal_shape, *g)
        return dual_problem(f, transform, separable_atom)

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
        out = self.separable_atom.nonsmooth_objective(x, 
                                                      check_feasibility=check_feasibility)
        if self.quadratic is None:
            return out
        else:
            return out + self.quadratic.objective(x, 'func')

    def proximal(self, proxq, prox_control=None):
        """
        The proximal function for the dual problem
        """
        transform, separable_atom = self.transform, self.separable_atom
        return separable_atom.proximal(proxq, prox_control)

def stacked_dual(shape, *primary_atoms):
    '''
    Computes a dual of 

    .. math::

       \sum_i g_i(D_i\beta)

    under the substitutions :math:`v_i=D_i\beta`.
    That is, it returns the following dual function after minimizing
    over :math:`(v_i,\beta_i)`:

    .. math::

       -\sum_i g_i^*(u_i)

    as well as the transform 
    :math:`D\maps \mathbb{R}^p \prod_i \mathbb{R}^{m_i}` where
    :math:`p` is the primal shape and :math:`m_i` are the corresponding
    dual shapes.

    Parameters
    ----------

    primary_atoms : [atoms]
        Objects that have dual attributes, which is a pair
        (ltransform, conjugate).

    '''
    if len(primary_atoms) == 0:
        primary_atoms = [zero_nonsmooth(shape)]

    duals = [atom.dual for atom in primary_atoms]
    transforms = [d[0] for d in duals]
    dual_atoms = [d[1] for d in duals]

    if len(transforms) > 1:            
        transform = afvstack(transforms)
        separable_atom = separable(transform.dual_shape, dual_atoms,
                                   transform.dual_slices)
        _dual = transform, separable_atom
    else:
        _dual = (transforms[0], dual_atoms[0])
    return _dual

