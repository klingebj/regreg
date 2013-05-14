from copy import copy
import warnings

import numpy as np

from .identity_quadratic import identity_quadratic
from .composite import nonsmooth
from .affine import (linear_transform, identity as identity_transform, 
                    affine_transform)
from .smooth import affine_smooth
from .objdoctemplates import objective_doc_templater
from .doctemplates import (doc_template_user, doc_template_provider)


@objective_doc_templater()
class atom(nonsmooth):

    """
    A class that defines the API for support functions.
    """
    tol = 1.0e-05

    def latexify(self, var='x', idx=''):
        template = {}
        if self.offset is None or np.all(self.offset == 0):
            template['var'] = var
        else:
            template['var'] = var + r'+\alpha_{%s}' % str(idx)

        obj = self.objective_template % template
        if self.lagrange is not None:
            obj = r'\lambda_{%s} %s' % (idx, obj)
        else:
            obj = r'I^{\infty}(%s \leq \epsilon_{%s})' % (obj, idx)

        if not self.quadratic.iszero:
            return ' + '.join([self.quadratic.latexify(var=var, idx=idx), obj])
        return obj

    def get_conjugate(self):
        """
        Return the conjugate of an given atom.
        Abstract method: subclasses must implement.
        """
        return None
    conjugate = property(get_conjugate)

    @property
    def dual(self):
        """
        Return the dual of an atom. This dual is formed by making the a substitution
        of the form v=Ax where A is the self.linear_transform.

        >>> penalty
        l1norm((30,), lagrange=2.300000, offset=None)
        >>> penalty.dual
        (<regreg.affine.identity object at 0x10a900bd0>, supnorm((30,), bound=2.300000, offset=0))

        If there is a linear part to the penalty, the linear_transform may not be identity:

        >>> D = (np.identity(4) + np.diag(-np.ones(3),1))[:-1]
        >>> D
        array([[ 1., -1.,  0.,  0.],
               [ 0.,  1., -1.,  0.],
               [ 0.,  0.,  1., -1.]])
        >>> fused_lasso = l1norm.linear(D, lagrange=2.3)
        >>> fused_lasso
        affine_atom(l1norm((3,), lagrange=2.300000, offset=None), array([[ 1., -1.,  0.,  0.],
               [ 0.,  1., -1.,  0.],
               [ 0.,  0.,  1., -1.]]))
        >>> fused_lasso.dual
        (<regreg.affine.linear_transform object at 0x10a760a50>, supnorm((3,), bound=2.300000, offset=0))
        >>> 


        """
        return self.linear_transform, self.conjugate

    @property
    def linear_transform(self):
        """
        The linear transform applied before a penalty is computed. Defaults to regreg.affine.identity

        >>> penalty = l1norm(30, lagrange=3.4)
        >>> penalty.linear_transform
        <regreg.affine.identity object at 0x10ae825d0>

        """
        if not hasattr(self, "_linear_transform"):
            self._linear_transform = identity_transform(self.primal_shape)
        return self._linear_transform

    @doc_template_provider
    def nonsmooth_objective(self, arg, check_feasibility=False):
        """
        The nonsmooth objective function of the atom.
        Includes the quadratic term of the atom.

        Abstract method: subclasses must implement.
        """
        raise NotImplementedError

    def smoothed(self, smoothing_quadratic):
        '''
        Add quadratic smoothing term
        '''
        conjugate_atom = copy(self.conjugate)
        sq = smoothing_quadratic
        if sq.coef in [None, 0]:
            raise ValueError('quadratic term of ' 
                             + 'smoothing_quadratic must be non 0')
        total_q = sq

        if conjugate_atom.quadratic is not None:
            total_q = sq + conjugate_atom.quadratic
        conjugate_atom.quadratic = total_q
        smoothed_atom = conjugate_atom.conjugate
        return smoothed_atom

    @doc_template_provider
    def proximal(self, proxq, prox_control=None):
        r"""
        The proximal operator. 
        Abstract method -- subclasses must implement.
        """
        raise NotImplementedError

class affine_atom(object):
    r"""
    Given a seminorm on :math:`\mathbb{R}^p`, i.e.  :math:`\beta \mapsto
    h_K(\beta)` this class creates a new seminorm that evaluates
    :math:`h_K(D\beta+\alpha)`

    This class does not have a prox, but its dual does. The prox of the dual is

    .. math::

       \text{minimize} \frac{1}{2} \|y-x\|^2_2 + x^T\alpha
       \ \text{s.t.} \ x \in \lambda K

    """

    def __init__(self, atom_obj, atransform):
        self.atom = copy(atom_obj)
        # if the affine transform has an offset, put it into
        # the atom. in this way, the affine_transform is actually
        # always linear
        if atransform.affine_offset is not None:
            if self.atom.offset is not None:
                self.atom.offset += atransform.affine_offset
            else:
                self.atom.offset = atransform.affine_offset
            ltransform = linear_transform(atransform.linear_operator,
                                          diag=atransform.diagD)
        else:
            ltransform = atransform
        self.linear_transform = ltransform
        self.primal_shape = self.linear_transform.primal_shape
        self.dual_shape = self.linear_transform.dual_shape

    def latexify(self, var='x', idx=''):
        return self.atom.latexify(var='D_{%s}%s' % (idx, var), idx=idx)

    def __repr__(self):
        return "affine_atom(%s, %s)" % (repr(self.atom),
                                        repr(self.linear_transform.linear_operator))

    @property
    def dual(self):
        tmpatom = copy(self.atom)
        tmpatom.primal_shape = tmpatom.dual_shape = self.dual_shape
        return self.linear_transform, tmpatom.conjugate

    def nonsmooth_objective(self, arg, check_feasibility=False):
        """
        Return self.atom.seminorm(self.linear_transform.linear_map(x))
        """
        return self.atom.nonsmooth_objective( \
            self.linear_transform.linear_map(arg),
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
            raise ValueError('quadratic term of '
                             + 'smoothing_quadratic must be non 0')
        conjugate_atom.quadratic = total_q
        smoothed_atom = conjugate_atom.conjugate
        value = affine_smooth(smoothed_atom, ltransform)
        value.total_quadratic = smoothed_atom.total_quadratic
        return value

def _work_out_conjugate(offset, quadratic):
    if offset is None:
        offset = 0
    else:
        offset = offset
    outq = identity_quadratic(0,0,-offset, \
          -quadratic.constant_term + 
          np.sum(offset * quadratic.linear_term))

    if quadratic.linear_term is not None:
        outoffset = -quadratic.linear_term
    else:
        outoffset = None
    return outoffset, outq

