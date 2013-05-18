import numpy as np
from scipy import sparse
from copy import copy
import warnings

from ..problems.composite import composite, nonsmooth
from .cones import cone, affine_cone
from ..identity_quadratic import identity_quadratic
from ..atoms import _work_out_conjugate

try:
    from projl1_cython import projl1
except:
    warnings.warn('Cython version of projl1 not available. Using slower python version')
    from projl1_python import projl1

class linear_constraint(cone):

    """
    This class allows specifications of linear constraints
    of the form :math:`x \in \text{row}(L)` by specifying
    an orthonormal basis for the rowspace of :math:`L`.

    If the constraint is of the form :math:`Ax=0`, then
    this linear constraint can be created using the
    *linear* classmethod of the *zero* cone in *regreg.cones*.
    
    """
    tol = 1.0e-05

    #XXX should basis by a linear operator instead?
    def __init__(self, shape, basis,
                 offset=None,
                 initial=None, 
                 quadratic=None):

        cone.__init__(self,
                      shape,
                      offset=offset,
                      initial=initial,
                      quadratic=quadratic)
        self.basis = np.asarray(basis)
        

    def __eq__(self, other):
        if self.__class__ == other.__class__:
            return (self.shape == other.shape
                    and np.allclose(other.basis, self.basis))
        return False

    def __copy__(self):
        return self.__class__(copy(self.shape),
                              self.basis.copy(),
                              offset=copy(self.offset),
                              quadratic=self.quadratic)
    
    def __repr__(self):
        if self.quadratic.iszero:
            return "%s(%s, %s, offset=%s)" % \
                (self.__class__.__name__,
                 repr(self.basis),
                 repr(self.shape), 
                 repr(self.offset))
        else:
            return "%s(%s, %s, offset=%s, quadratic=%s)" % \
                (self.__class__.__name__,
                 repr(self.basis),
                 repr(self.shape), 
                 repr(self.offset),
                 self.quadratic)

    @property
    def conjugate(self):
        if self.quadratic.coef == 0:

            offset, outq = _work_out_conjugate(self.offset, self.quadratic)

            cls = conjugate_cone_pairs[self.__class__]
            atom = cls(self.shape, 
                       self.basis,
                       offset=offset,
                       quadratic=outq)
        else:
            atom = smooth_conjugate(self)
        self._conjugate = atom
        self._conjugate._conjugate = self
        return self._conjugate

    @classmethod
    def linear(cls, linear_operator, basis, diag=False,
               linear_term=None, offset=None):
        l = linear_transform(linear_operator, diag=diag)
        cone = cls(l.input_shape, basis,
                   linear_term=linear_term, offset=offset)
        return affine_cone(cone, l)

class projection(linear_constraint):

    """
    An atom representing a linear constraint.
    It is specified via a matrix that is assumed
    to be an set of row vectors spanning the space.

    Notes
    =====

    It is assumed (without checking) that the rows of basis
    are orthonormal, so that projecting *x* onto their span
    is simply np.dot(basis.T, np.dot(basis, x))
    """

    def constraint(self, x):
        """
        The non-positive constraint of x.
        """
        projx = self.cone_prox(x)
        incone = np.linalg.norm(x-projx) / max([np.linalg.norm(x),1]) < self.tol
        if incone:
            return 0
        return np.inf

    def cone_prox(self, x,  lipschitz=1):
        r"""
        Return (unique) minimizer

        .. math::

            v^{\lambda}(x) = \text{argmin}_{v \in \mathbb{R}^p} \frac{L}{2}
            \|x-v\|^2_2  \; \text{ s.t.} \; x \in \text{row}(L)

        where $p$=x.shape[0], :math:`\lambda` = self.lagrange and self.basis is
        an orthonormal basis for :math:`\text{row}(L)`

        This is just projection onto :math:`\text{row}(L)`.
        """
        coefs = np.dot(self.basis, x)
        return np.dot(coefs, self.basis)


class projection_complement(linear_constraint):

    """
    An atom representing a linear constraint.
    The orthogonal complement of projection, it is specified
    with an orthonormal basis for the complement
    """

    def constraint(self, x):
        """
        The non-positive constraint of x.
        """
        projx = self.cone_prox(x)
        incone = np.linalg.norm(projx) / max([np.linalg.norm(x),1]) < self.tol
        if incone:
            return 0
        return np.inf

    def cone_prox(self, x,  lipschitz=1):
        r"""
        Return (unique) minimizer

        .. math::

            v^{\lambda}(x) = \text{argmin}_{v \in \mathbb{R}^p} \frac{L}{2}
            \|x-v\|^2_2  \; \text{ s.t.} \; Lx=0

        where $p$ = x.shape[0], :math:`\lambda` = self.lagrange and self.basis
        is an orthonormal basis for the orthogonal complement of
        :math:`\text{row}(L)`

        This is just projection onto the orthogonal complement of
        :math:`\text{row}(L)`
        """
        coefs = np.dot(self.basis, x)
        return x - np.dot(coefs, self.basis)

conjugate_cone_pairs = {}
for n1, n2 in [(projection, projection_complement),
               ]:
    conjugate_cone_pairs[n1] = n2
    conjugate_cone_pairs[n2] = n1
