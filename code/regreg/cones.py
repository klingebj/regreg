from copy import copy
import warnings

from scipy import sparse
import numpy as np

from .composite import composite, nonsmooth
from .affine import linear_transform, identity as identity_transform
from .identity_quadratic import identity_quadratic
from .atoms import smooth_conjugate

try:
    from projl1_cython import projl1
except:
    warnings.warn('Cython version of projl1 not available. Using slower python version')
    from projl1_python import projl1



class cone(nonsmooth):

    """
    A class that defines the API for cone constraints.
    """
    tol = 1.0e-05

    def __init__(self, primal_shape,
                 offset=None,
                 initial=None,
                 quadratic=None):

        self.offset = None
        if offset is not None:
            self.offset = np.array(offset)

        if type(primal_shape) == type(1):
            self.primal_shape = (primal_shape,)
        else:
            self.primal_shape = primal_shape
        self.dual_shape = self.primal_shape

        if initial is None:
            self.coefs = np.zeros(self.primal_shape)
        else:
            self.coefs = initial.copy()

        if quadratic is not None:
            self.set_quadratic(quadratic.coef, quadratic.offset,
                               quadratic.linear_term, 
                               quadratic.constant_term)
        else:
            self.set_quadratic(0,0,0,0)

    def __eq__(self, other):
        if self.__class__ == other.__class__:
            return self.primal_shape == other.primal_shape
        return False

    def __copy__(self):
        return self.__class__(copy(self.primal_shape),
                              offset=copy(self.offset),
                              initial=self.coefs,
                              quadratic=self.quadratic)
    
    def __repr__(self):
        if self.quadratic is None or not self.quadratic.anything_to_return:
            return "%s(%s, offset=%s)" % \
                (self.__class__.__name__,
                 `self.primal_shape`, 
                 str(self.offset))
        else:
            return "%s(%s, offset=%s, quadratic=%s)" % \
                (self.__class__.__name__,
                 `self.primal_shape`, 
                 str(self.offset),
                 str(self.quadratic))

    @property
    def conjugate(self):
        if self.quadratic.coef == 0:
            if self.offset is not None:
                if self.quadratic.linear_term is not None:
                    outq = identity_quadratic(0, None, -self.offset, -self.quadratic.constant_term + (self.offset * self.quadratic.linear_term).sum())
                else:
                    outq = identity_quadratic(0, None, -self.offset, -self.quadratic.constant_term)
            else:
                outq = identity_quadratic(0, 0, 0, -self.quadratic.constant_term)
            if self.quadratic.linear_term is not None:
                offset = -self.quadratic.linear_term
            else:
                offset = None

            cls = conjugate_cone_pairs[self.__class__]
            atom = cls(self.primal_shape, 
                       offset=offset,
                       quadratic=outq)
        else:
            atom = smooth_conjugate(self)
        self._conjugate = atom
        self._conjugate._conjugate = self
        return self._conjugate

    @property
    def dual(self):
        return self.linear_transform, self.conjugate

    @property
    def linear_transform(self):
        if not hasattr(self, "_linear_transform"):
            self._linear_transform = identity_transform(self.primal_shape)
        return self._linear_transform
    
    def constraint(self, x):
        """
        Abstract method. Evaluate the constraint on the dual norm of x.
        """
        raise NotImplementedError

    def nonsmooth_objective(self, x, check_feasibility=False):
        if self.offset is not None:
            x_offset = x + self.offset
        else:
            x_offset = x
        if check_feasibility:
            v = self.constraint(x_offset)
        else:
            v = 0
        v += self.quadratic.objective(x, 'func')
        return v
        
    def proximal(self, lipschitz, x, grad):
        r"""
        The proximal operator. If the atom is in
        Lagrange mode, this has the form

        .. math::

           v^{\lambda}(x) = \text{argmin}_{v \in \mathbb{R}^p} \frac{L}{2}
           \|x-v\|^2_2 + \lambda h(v+\alpha) + \langle v, \eta \rangle

        where :math:`\alpha` is the offset of self.affine_transform and
        :math:`\eta` is self.linear_term.

        .. math::

           v^{\lambda}(x) = \text{argmin}_{v \in \mathbb{R}^p} \frac{L}{2}
           \|x-v\|^2_2 + \langle v, \eta \rangle \text{s.t.} \   h(v+\alpha) \leq \lambda

        """

        proxq = identity_quadratic(lipschitz, -x, grad)
        totalq = self.quadratic + proxq

        if self.offset is None or np.all(np.equal(self.offset, 0)):
            offset = None
        else:
            offset = self.offset

        if offset is not None:
            totalq.offset = -offset
            totalq = totalq.collapsed()

        if totalq.coef == 0:
            raise ValueError('lipschitz + quadratic coef must be positive')

        prox_arg = -totalq.linear_term / totalq.coef

        debug = False
        if debug:
            print '='*80
            print 'x :', x
            print 'grad: ', grad
            print 'cone: ', self
            print 'proxq: ', proxq
            print 'proxarg: ', prox_arg
            print 'totalq: ', totalq

        eta = self.cone_prox(prox_arg, lipschitz=lipschitz)
        if offset is None:
            return eta
        else:
            return eta - offset

    def cone_prox(self, x, lipschitz=1):
        r"""
        Return (unique) minimizer

        .. math::

           v^{\lambda}(x) = \text{argmin}_{v \in \mathbb{R}^p} \frac{L}{2}
           \|x-v\|^2_2 + \lambda h(v)

        where *p*=x.shape[0] and :math:`h(v)` is the support function of self (with a
        Lagrange multiplier of 1 in front) and :math:`\lambda` is the Lagrange parameter.
        If the argument is None and the atom is in Lagrange mode, this parameter
        is used for the proximal operator, else an exception is raised.
        
        """
        raise NotImplementedError

    @classmethod
    def linear(cls, linear_operator, diag=False,
               offset=None,
               quadratic=None):
        l = linear_transform(linear_operator, diag=diag)
        cone = cls(l.primal_shape, 
                   offset=offset,
                   quadratic=quadratic)
        return affine_cone(cone, l)

class affine_cone(object):

    """
    Given a seminorm on :math:`\mathbb{R}^p`, i.e.
    :math:`\beta \mapsto h_K(\beta)`
    this class creates a new seminorm 
    that evaluates :math:`h_K(D\beta+\alpha)`

    This class does not have a prox, but its
    dual does. The prox of the dual is

    .. math::

       \text{minimize} \frac{1}{2} \|y-x\|^2_2 + x^T\alpha
       \ \text{s.t.} \ x \in \lambda K
    
    """

    def __init__(self, cone_obj, atransform):
        self.cone = copy(cone_obj)
        # if the affine transform has an offset, put it into
        # the cone. in this way, the affine_transform is actually
        # always linear
        if atransform.affine_offset is not None:
            self.cone.offset += atransform.affine_offset
            ltransform = affine_transform(atransform.linear_operator, None,
                                          diag=atransform.diag)
        else:
            ltransform = atransform
        self.linear_transform = ltransform
        self.primal_shape = self.linear_transform.primal_shape
        self.dual_shape = self.linear_transform.dual_shape

    def __repr__(self):
        return "affine_cone(%s, %s)" % (`self.cone`,
                                        `self.linear_transform.linear_operator`)

    @property
    def dual(self):
        return self.linear_transform, self.cone.conjugate

    def nonsmooth_objective(self, x, check_feasibility=False):
        """
        Return self.cone.nonsmooth_objective(self.linear_transform.linear_map(x))
        """
        return self.cone.nonsmooth_objective(self.linear_transform.linear_map(x),
                                             check_feasibility=check_feasibility)


class nonnegative(cone):

    """
    The non-negative cone constraint (which is the support
    function of the non-positive cone constraint).
    """
    
    def constraint(self, x):
        """
        The non-negative constraint of x.
        """
        tol_lim = np.fabs(x).max() * self.tol
        incone = np.all(np.greater_equal(x, -tol_lim))
        if incone:
            return 0
        return np.inf


    def cone_prox(self, x,  lipschitz=1):
        r"""
        Return (unique) minimizer

        .. math::

            v^{\lambda}(x) = \text{argmin}_{v \in \mathbb{R}^p} \frac{L}{2}
            \|x-v\|^2_2 \ \text{s.t.} \  v_i \geq 0.

        where *p*=x.shape[0], :math:`\lambda` = self.lagrange. 
        This is just a element-wise
        np.maximum(x, 0)

        .. math::

            v^{\lambda}(x)_i = \max(x_i, 0)

        """
        return np.maximum(x, 0)


class nonpositive(nonnegative):

    """
    The non-positive cone constraint (which is the support
    function of the non-negative cone constraint).
    """
    
    def constraint(self, x):
        """
        The non-positive constraint of x.
        """
        tol_lim = np.fabs(x).max() * self.tol
        incone = np.all(np.less_equal(x, tol_lim))
        if incone:
            return 0
        return np.inf

    def cone_prox(self, x,  lipschitz=1):
        r"""
        Return unique minimizer

        .. math::

            v^{\lambda}(x) = \text{argmin}_{v \in \mathbb{R}^p} \frac{L}{2}
            \|x-v\|^2_2 \ \text{s.t.} \  v_i \leq 0.

        where *p*=x.shape[0], :math:`\lambda` = self.lagrange. 
        This is just a element-wise
        np.maximum(x, 0)

        .. math::

            v^{\lambda}(x)_i = \min(x_i, 0)

        """
        return np.minimum(x, 0)


class zero(cone):
    """
    The zero seminorm, support function of :math:\{0\}
    """

    def constraint(self, x):
        return 0.

    def cone_prox(self, x, lipschitz=1):
        return x

class zero_constraint(cone):
    """
    The zero constraint, support function of :math:`\mathbb{R}`^p
    """

    def constraint(self, x):
        if not np.linalg.norm(x) <= self.tol:
            return np.inf
        return 0.

    def cone_prox(self, x, lipschitz=1):
        return np.zeros(np.asarray(x).shape)


conjugate_cone_pairs = {}
for n1, n2 in [(nonnegative,nonpositive),
               (zero, zero_constraint)
               ]:
    conjugate_cone_pairs[n1] = n2
    conjugate_cone_pairs[n2] = n1
