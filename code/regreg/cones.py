from copy import copy
import warnings

from scipy import sparse
import numpy as np

from .composite import composite, nonsmooth, smooth_conjugate
from .affine import linear_transform, identity as identity_transform
from .identity_quadratic import identity_quadratic
from .atoms import _work_out_conjugate
from .smooth import affine_smooth

try:
    from projl1_cython import projl1_epigraph, projlinf_epigraph
except:
    warnings.warn('Cython version of projl1 not available. Using slower python version')
    from projl1_python import projl1

class cone(nonsmooth):

    _doc_dict = {'linear':r' + \langle \eta, x \rangle',
                 'constant':r' + \tau',
                 'objective': '',
                 'shape':'p',
                 'var':r'x'}

    """
    A class that defines the API for cone constraints.
    """
    tol = 1.0e-05

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
        if self.quadratic.iszero:
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
            offset, outq = _work_out_conjugate(self.offset, 
                                               self.quadratic)
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

    def latexify(self, var='x', idx=''):
        d = {}
        if self.offset is None:
            d['var'] = var
        else:
            d['var'] = var + r'+\alpha_{%s}' % str(idx)

        obj = self.objective_template % d

        if not self.quadratic.iszero:
            return ' + '.join([self.quadratic.latexify(var=var,idx=idx),obj])
        return obj

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

    def proximal(self, proxq, prox_control=None):
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
        offset, totalq = (self.quadratic + proxq).recenter(self.offset)
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

        eta = self.cone_prox(prox_arg, lipschitz=totalq.coef)
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

        where *p*=x.shape[0] and :math:`h(v)` is the support function of self
        (with a Lagrange multiplier of 1 in front) and :math:`\lambda` is the
        Lagrange parameter.  If the argument is None and the atom is in Lagrange
        mode, this parameter is used for the proximal operator, else an
        exception is raised.
        """
        raise NotImplementedError

    @classmethod
    def linear(cls, linear_operator, diag=False,
               offset=None,
               quadratic=None):
        if not isinstance(linear_operator, linear_transform):
            l = linear_transform(linear_operator, diag=diag)
        else:
            l = linear_operator
        cone = cls(l.dual_shape, 
                   offset=offset,
                   quadratic=quadratic)
        return affine_cone(cone, l)

    @classmethod
    def affine(cls, linear_operator, offset, diag=False,
               quadratic=None):
        if not isinstance(linear_operator, linear_transform):
            l = linear_transform(linear_operator, diag=diag)
        else:
            l = linear_operator
        cone = cls(l.dual_shape, 
                   offset=offset,
                   quadratic=quadratic)
        return affine_cone(cone, l)


class affine_cone(object):
    r"""
    Given a seminorm on :math:`\mathbb{R}^p`, i.e.  :math:`\beta \mapsto
    h_K(\beta)` this class creates a new seminorm that evaluates
    :math:`h_K(D\beta+\alpha)`

    This class does not have a prox, but its dual does. The prox of the dual is

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

    def latexify(self, var='x', idx=''):
        return self.cone.latexify(var='D_{%s}%s' % (idx, x), idx=idx)

    @property
    def dual(self):
        return self.linear_transform, self.cone.conjugate

    def nonsmooth_objective(self, x, check_feasibility=False):
        """
        Return self.cone.nonsmooth_objective(self.linear_transform.linear_map(x))
        """
        return self.cone.nonsmooth_objective(self.linear_transform.linear_map(x),
                                             check_feasibility=check_feasibility)

    def smoothed(self, smoothing_quadratic):
        '''
        Add quadratic smoothing term
        '''
        ltransform, conjugate_atom = self.dual
        if conjugate_atom.quadratic is not None:
            total_q = smoothing_quadratic + conjugate_atom.quadratic
        else:
            total_q = smoothing_quadratic
        if total_q.coef in [None, 0]:
            raise ValueError('quadratic term of smoothing_quadratic must be non 0')
        conjugate_atom.quadratic = total_q
        smoothed_atom = conjugate_atom.conjugate
        return affine_smooth(smoothed_atom, ltransform)


class nonnegative(cone):
    """
    The non-negative cone constraint (which is the support
    function of the non-positive cone constraint).
    """

    objective_template = r"""I^{\infty}(%(var)s \succeq 0)"""
    _doc_dict = copy(cone._doc_dict)
    _doc_dict['objective'] = objective_template % {'var': r'x + \alpha'}

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

    objective_template = r"""I^{\infty}(%(var)s \preceq 0)"""
    _doc_dict = copy(cone._doc_dict)
    _doc_dict['objective'] = objective_template % {'var': r'x + \alpha'}

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

    objective_template = r"""{\cal Z}(%(var)s)"""
    _doc_dict = copy(cone._doc_dict)
    _doc_dict['objective'] = objective_template % {'var': r'x + \alpha'}


    def constraint(self, x):
        return 0.

    def cone_prox(self, x, lipschitz=1):
        return x

class zero_constraint(cone):
    """
    The zero constraint, support function of :math:`\mathbb{R}`^p
    """

    objective_template = r"""I^{\infty}(%(var)s = 0)"""
    _doc_dict = copy(cone._doc_dict)
    _doc_dict['objective'] = objective_template % {'var': r'x + \alpha'}

    def constraint(self, x):
        if not np.linalg.norm(x) <= self.tol:
            return np.inf
        return 0.

    def cone_prox(self, x, lipschitz=1):
        return np.zeros(np.asarray(x).shape)

class l2_epigraph(cone):

    """
    The l2_epigraph constraint.
    """

    objective_template = r"""I^{\infty}(%(var)s \in \mathbf{epi}(\ell_2)})"""
    _doc_dict = copy(cone._doc_dict)
    _doc_dict['objective'] = objective_template % {'var': r'x + \alpha'}

    def constraint(self, x):
        """
        The non-negative constraint of x.
        """
        
        incone = np.linalg.norm(x[1:]) / x[0] <= 1 + self.tol
        if incone:
            return 0
        return np.inf


    def cone_prox(self, x,  lipschitz=1):
        r"""
        Return (unique) minimizer

        .. math::

            v^{\lambda}(x) = \text{argmin}_{v \in \mathbb{R}^p} \frac{L}{2}
            \|x-v\|^2_2 \ \text{s.t.} \  v \in \mathbf{epi}(\ell_2)

        where *p*=x.shape[0], :math:`\lambda` = self.lagrange. 

        """
        norm = x[0]
        coef = x[1:]
        norm_coef = np.linalg.norm(coef)
        thold = (norm_coef - norm) / 2.
        result = np.zeros_like(x)
        result[1:] = coef / norm_coef * max(norm_coef - thold, 0)
        result[0] = max(norm + thold, 0)
        return result

class l2_epigraph_polar(cone):

    """
    The polar of the l2_epigraph constraint, which is the negative of the 
    l2 epigraph..
    """

    objective_template = r"""I^{\infty}(-%(var)s \in \mathbf{epi}(\ell_2)})"""
    _doc_dict = copy(cone._doc_dict)
    _doc_dict['objective'] = objective_template % {'var': r'x + \alpha'}

    def constraint(self, x):
        """
        The non-negative constraint of x.
        """
        incone = np.linalg.norm(x[1:]) / -x[0] <= 1 + self.tol
        if incone:
            return 0
        return np.inf


    def cone_prox(self, x,  lipschitz=1):
        r"""
        Return (unique) minimizer

        .. math::

            v^{\lambda}(x) = \text{argmin}_{v \in \mathbb{R}^p} \frac{L}{2}
            \|x-v\|^2_2 \ \text{s.t.} \  -v \in \mathbf{epi}(\ell_2)

        where *p*=x.shape[0], :math:`\lambda` = self.lagrange. 

        """
        norm = -x[0]
        coef = -x[1:]
        norm_coef = np.linalg.norm(coef)
        thold = (norm_coef - norm) / 2.
        result = np.zeros_like(x)
        result[1:] = coef / norm_coef * max(norm_coef - thold, 0)
        result[0] = max(norm + thold, 0)
        return x + result


class l1_epigraph(cone):

    """
    The l1_epigraph constraint.
    """

    objective_template = r"""I^{\infty}(%(var)s \in \mathbf{epi}(\ell_1)})"""
    _doc_dict = copy(cone._doc_dict)
    _doc_dict['objective'] = objective_template % {'var': r'x + \alpha'}

    def constraint(self, x):
        """
        The non-negative constraint of x.
        """
        incone = np.fabs(x[1:]).sum() / x[0] <= 1 + self.tol
        if incone:
            return 0
        return np.inf


    def cone_prox(self, x,  lipschitz=1):
        r"""
        Return (unique) minimizer

        .. math::

            v^{\lambda}(x) = \text{argmin}_{v \in \mathbb{R}^p} \frac{L}{2}
            \|x-v\|^2_2 \ \text{s.t.} \  v \in \mathbf{epi}(\ell_1)

        where *p*=x.shape[0], :math:`\lambda` = self.lagrange. 

        """
        return projl1_epigraph(x)

class l1_epigraph_polar(cone):

    """
    The polar l1_epigraph constraint which is just the
    negative of the linf_epigraph.
    """

    objective_template = r"""I^{\infty}(%(var)s  \in -\mathbf{epi}(\ell_{\inf})})"""
    _doc_dict = copy(cone._doc_dict)
    _doc_dict['objective'] = objective_template % {'var': r'x + \alpha'}

    def constraint(self, x):
        """
        The non-negative constraint of x.
        """
        
        incone = np.fabs(-x[1:]).max() / -x[0] <= 1 + self.tol
        if incone:
            return 0
        return np.inf


    def cone_prox(self, x,  lipschitz=1):
        r"""
        Return (unique) minimizer

        .. math::

            v^{\lambda}(x) = \text{argmin}_{v \in \mathbb{R}^p} \frac{L}{2}
            \|x-v\|^2_2 \ \text{s.t.} \  -v \in \mathbf{epi}(\ell_{\inf})

        where *p*=x.shape[0], :math:`\lambda` = self.lagrange. 

        """
        return -projlinf_epigraph(-x)


class linf_epigraph(cone):

    """
    The l1_epigraph constraint.
    """

    objective_template = r"""I^{\infty}(%(var)s \in \mathbf{epi}(\ell_1)})"""
    _doc_dict = copy(cone._doc_dict)
    _doc_dict['objective'] = objective_template % {'var': r'x + \alpha'}

    def constraint(self, x):
        """
        The non-negative constraint of x.
        """

        incone = np.fabs(x[1:]).max() / x[0] <= 1 + self.tol
        if incone:
            return 0
        return np.inf


    def cone_prox(self, x,  lipschitz=1):
        r"""
        Return (unique) minimizer

        .. math::

            v^{\lambda}(x) = \text{argmin}_{v \in \mathbb{R}^p} \frac{L}{2}
            \|x-v\|^2_2 \ \text{s.t.} \  v \in \mathbf{epi}(\ell_1)

        where *p*=x.shape[0], :math:`\lambda` = self.lagrange. 

        """
        return projlinf_epigraph(x)


class linf_epigraph_polar(cone):

    """
    The polar linf_epigraph constraint which is just the
    negative of the l1_epigraph.
    """

    objective_template = r"""I^{\infty}(%(var)s \in -\mathbf{epi}(\ell_1)})"""
    _doc_dict = copy(cone._doc_dict)
    _doc_dict['objective'] = objective_template % {'var': r'x + \alpha'}

    def constraint(self, x):
        """
        The non-negative constraint of x.
        """
        
        incone = np.fabs(-x[1:]).max() / -x[0] <= 1 + self.tol
        if incone:
            return 0
        return np.inf


    def cone_prox(self, x,  lipschitz=1):
        r"""
        Return (unique) minimizer

        .. math::

            v^{\lambda}(x) = \text{argmin}_{v \in \mathbb{R}^p} \frac{L}{2}
            \|x-v\|^2_2 \ \text{s.t.} \  -v \in \mathbf{epi}(\ell_{\inf})

        where *p*=x.shape[0], :math:`\lambda` = self.lagrange. 

        """
        return -projl1_epigraph(-x)


conjugate_cone_pairs = {}
for n1, n2 in [(nonnegative,nonpositive),
               (zero, zero_constraint),
               (l1_epigraph, l1_epigraph_polar),
               (l2_epigraph, l2_epigraph_polar),
               (linf_epigraph, linf_epigraph_polar)
               ]:
    conjugate_cone_pairs[n1] = n2
    conjugate_cone_pairs[n2] = n1
