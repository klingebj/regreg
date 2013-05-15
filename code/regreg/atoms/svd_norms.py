"""
This module contains the implementation operator and nuclear norms, used in 
matrix completion problems and other low-rank factorization
problems.

"""
from md5 import md5
from copy import copy

import numpy as np

from ..atoms import atom, _work_out_conjugate
from .seminorms import conjugate_seminorm_pairs, seminorm
from .cones import cone, conjugate_cone_pairs
from .projl1_cython import projl1, projl1_epigraph

from ..objdoctemplates import objective_doc_templater
from ..doctemplates import (doc_template_user, doc_template_provider)


class svd_obj(object):

    def compute_and_store_svd(self, X):
        """
        Compute and store svd of X for use in multiple function calls.
        """
        self._md5x = md5(X).hexdigest()
        self._X = X
        self.SVD = np.linalg.svd(X, full_matrices=0)
        return self.SVD

    def setX(self, X):
        if not hasattr(self, "_md5x") or md5(X).hexdigest() != self._md5x:
            self.compute_and_store_svd(X)
    def getX(self):
        if hasattr(self, "_X"):
            return self._X
        raise AttributeError("X has not been set")
    X = property(getX, setX)

    def get_conjugate(self):
        seminorm.get_conjugate(self)
        # share the SVD with the conjugate
        if hasattr(self, "_X"):
            for attr in ['_X', '_U' ,'_D', '_V']:
                setattr(self._conjugate, attr, getattr(self, attr))
        return self._conjugate
    conjugate = property(get_conjugate)

    def get_SVD(self):
        if hasattr(self, "_U"):
            return self._U, self._D, self._V
        raise AttributeError("SVD has not been set")
    def set_SVD(self, UDV):
        self._U, self._D, self._V = UDV
    SVD = property(get_SVD, set_SVD)

@objective_doc_templater()
class svd_atom(seminorm, svd_obj):

    _doc_dict = {'linear':r' + \text{Tr}(\eta^T X)',
                 'constant':r' + \tau',
                 'objective': '',
                 'shape':r'p \times q',
                 'var':r'X'}

    @doc_template_provider
    def lagrange_prox(self, X, lipschitz=1, lagrange=None):
        r""" Return unique minimizer

        .. math::

           %(var)s^{\lambda}(U) =
           \text{argmin}_{%(var)s \in \mathbb{R}^{%(shape)s}} 
           \frac{L}{2}
           \|U-%(var)s\|^2_F %(linear)s %(constant)s \ 
            + \lambda   %(objective)s 

        Above, :math:`\lambda` is the Lagrange parameter,
        :math:`A` is self.offset (if any), 
        :math:`\eta` is self.linear_term (if any)
        and :math:`\tau` is self.constant_term.

        If the argument `lagrange` is None and the atom is in lagrange mode,
        self.lagrange is used as the lagrange parameter, else an exception is
        raised.

        The class atom's lagrange_prox just returns the appropriate lagrange
        parameter for use by the subclasses.
        """
        if lagrange is None:
            lagrange = self.lagrange
        if lagrange is None:
            raise ValueError('either atom must be in Lagrange mode or a keyword "lagrange" argument must be supplied')
        return lagrange

    @doc_template_provider
    def bound_prox(self, X, lipschitz=1, bound=None):
        r"""
        Return unique minimizer

        .. math::

           %(var)s^{\lambda}(U) \in 
           \text{argmin}_{%(var)s \in \mathbb{R}^{%(shape)s}} 
           \frac{L}{2}
           \|U-%(var)s\|^2_F %(linear)s %(constant)s \ 
           \text{s.t.} \   %(objective)s \leq \lambda

        Above, :math:`\lambda` is the bound parameter,
        :math:`A` is self.offset (if any), 
        :math:`\eta` is self.linear_term (if any)
        and :math:`\tau` is self.constant_term (if any).

        If the argument `bound` is None and the atom is in bound mode,
        self.bound is used as the bound parameter, else an exception is raised.

        The class atom's bound_prox just returns the appropriate bound
        parameter for use by the subclasses.
        """
        if bound is None:
            bound = self.bound
        if bound is None:
            raise ValueError('either atom must be in bound mode or a keyword "bound" argument must be supplied')
        return bound


@objective_doc_templater()
class nuclear_norm(svd_atom):

    """
    The nuclear norm
    """
    prox_tol = 1.0e-10

    objective_template = r"""\|%(var)s\|_*"""
    objective_vars = {'var': r'X + A'}

    @doc_template_user
    def seminorm(self, X, check_feasibility=False,
                 lagrange=None):
        # This will compute an svd of X
        # if the md5 hash of X doesn't match.
        lagrange = seminorm.seminorm(self, X, lagrange=lagrange,
                                 check_feasibility=check_feasibility)
        self.X = X
        _, D, _ = self.SVD
        return self.lagrange * np.sum(D)

    @doc_template_user
    def constraint(self, X, bound=None):
        # This will compute an svd of X
        # if the md5 hash of X doesn't match.
        bound = seminorm.constraint(self, X, bound=bound)
        self.X = X
        _, D, _ = self.SVD
        inbox = np.sum(D) <= bound * (1 + self.tol)
        if inbox:
            return 0
        else:
            return np.inf

    @doc_template_user
    def lagrange_prox(self, X,  lipschitz=1, lagrange=None):
        lagrange = svd_atom.lagrange_prox(self, X, lipschitz, lagrange)
        self.X = X
        U, D, V = self.SVD
        D_soft_thresholded = np.maximum(D - lagrange/lipschitz, 0)
        keepD = D_soft_thresholded > 0
        self.SVD = U[:,keepD], D_soft_thresholded[keepD], V[keepD]
        c = self.conjugate
        c.SVD = self.SVD
        self._X = np.dot(U[:,keepD], D_soft_thresholded[keepD][:,np.newaxis] * V[keepD])
        return self.X

    @doc_template_user
    def bound_prox(self, X, lipschitz=1, bound=None):
        bound = svd_atom.bound_prox(self, X, lipschitz, bound)
        self.X = X
        U, D, V = self.SVD
        D_projected = projl1(D, bound)
        keepD = D_projected > 0
        # store the projected X -- or should we keep original?
        self.SVD = U[:,keepD], D_projected[keepD], V[keepD]
        c = self.conjugate
        c.SVD = self.SVD
        self._X = np.dot(U[:,keepD], D_projected[keepD][:,np.newaxis] * V[keepD])
        return self.X


@objective_doc_templater()
class operator_norm(svd_atom):

    """
    The operator norm
    """
    prox_tol = 1.0e-10

    objective_template = r"""\|%(var)s\|_{\text{op}}"""
    objective_vars = {'var': r'X + A'}

    @doc_template_user
    def seminorm(self, X, lagrange=None, check_feasibility=False):
        # This will compute an svd of X
        # if the md5 hash of X doesn't match.
        lagrange = seminorm.seminorm(self, X, lagrange=lagrange,
                                 check_feasibility=check_feasibility)
        self.X = X
        _, D, _ = self.SVD
        return self.lagrange * np.max(D)

    @doc_template_user
    def constraint(self, X, bound=None):
        # This will compute an svd of X
        # if the md5 hash of X doesn't match.
        bound = seminorm.constraint(self, X, bound=bound)
        self.X = X
        _, D, _ = self.SVD
        inbox = np.max(D) <= self.bound * (1 + self.tol)
        if inbox:
            return 0
        else:
            return np.inf

    @doc_template_user
    def lagrange_prox(self, X,  lipschitz=1, lagrange=None):
        lagrange = svd_atom.lagrange_prox(self, X, lipschitz, lagrange)
        self.X = X
        U, D, V = self.SVD
        D_soft_thresholded = D - projl1(D, lagrange/lipschitz)
        keepD = D_soft_thresholded > 0
        self.SVD = U[:,keepD], D_soft_thresholded[keepD], V[keepD]
        c = self.conjugate
        c.SVD = self.SVD
        self._X = np.dot(U[:,keepD], D_soft_thresholded[keepD][:,np.newaxis] * V[keepD])
        return self.X

    @doc_template_user
    def bound_prox(self, X, lipschitz=1, bound=None):
        bound = svd_atom.bound_prox(self, X, lipschitz, bound)
        self.X = X
        U, D, V = self.SVD
        self._D = np.minimum(D, bound)
        # store the projected X -- or should we keep original?
        self._X = np.dot(U, D[:,np.newaxis] * V)
        return self.X

#@objective_doc_templater()
class svd_cone(cone, svd_obj):

    def __repr__(self):
        if self.quadratic.iszero:
            return "%s(%s, offset=%s)" % \
                (self.__class__.__name__,
                 `self.matrix_shape`,
                 str(self.offset))
        else:
            return "%s(%s, offset=%s, quadratic=%s)" % \
                (self.__class__.__name__,
                 `self.matrix_shape`,
                 str(self.offset),
                 str(self.quadratic))

    def __init__(self, primal_shape,
                 offset=None,
                 quadratic=None,
                 initial=None):

        self.matrix_shape = primal_shape
        primal_shape = np.product(primal_shape)+1
        cone.__init__(self, primal_shape, offset=offset,
                      quadratic=quadratic,
                      initial=initial)

    @property
    def conjugate(self):
        if self.quadratic.coef == 0:
            offset, outq = _work_out_conjugate(self.offset, 
                                               self.quadratic)
            cls = conjugate_cone_pairs[self.__class__]
            print self.matrix_shape, 'matrixshape'
            new_atom = cls(self.matrix_shape,
                       offset=offset,
                       quadratic=outq)
        else:
            new_atom = smooth_conjugate(self)
        self._conjugate = new_atom
        self._conjugate._conjugate = self
        return self._conjugate


#@objective_doc_templater()
class nuclear_norm_epigraph(svd_cone):

    def constraint(self, normX):
        """
        The non-negative constraint of x.
        """
        norm = normX[0]
        X = normX[1:].reshape(self.matrix_shape)
        self.X = X
        U, D, V = self.SVD

        incone = np.fabs(D[1:]).sum() / norm <= 1 + self.tol
        if incone:
            return 0
        return np.inf

    def cone_prox(self, normX,  lipschitz=1):
        r"""
        Return (unique) minimizer

        .. math::

            v^{\lambda}(x) = \text{argmin}_{v \in \mathbb{R}^p} \frac{L}{2}
            \|x-v\|^2_2 \ \text{s.t.} \  v \in \mathbf{epi}(\ell_1)

        where *p*=x.shape[0], :math:`\lambda` = self.lagrange. 

        """
        norm = normX[0]
        X = normX[1:].reshape(self.matrix_shape)
        self.X = X
        U, D, V = self.SVD
        newD = np.zeros(D.shape[0]+1)
        newD[0] = norm
        newD[1:] = D
        newD = projl1_epigraph(newD)
        result = np.zeros_like(normX)
        result[0] = newD[0] 
        self.X = np.dot(U, newD[1:,np.newaxis] * V)
        result[1:] = self.X.reshape(-1)
        return result

#@objective_doc_templater()
class nuclear_norm_epigraph_polar(svd_cone):
    

    def constraint(self, normX):
        """
        The non-negative constraint of x.
        """
        
        norm = normX[0]
        X = normX[1:].reshape(self.matrix_shape)
        self.X = X
        U, D, V = self.SVD

        incone = D.max() / -norm <= 1 + self.tol
        if incone:
            return 0
        return np.inf


    def cone_prox(self, normX,  lipschitz=1):
        r"""
        Return (unique) minimizer

        .. math::

            v^{\lambda}(x) = \text{argmin}_{v \in \mathbb{R}^p} \frac{L}{2}
            \|x-v\|^2_2 \ \text{s.t.} \  v \in \mathbf{epi}(\ell_1)

        where *p*=x.shape[0], :math:`\lambda` = self.lagrange. 

        """
        norm = normX[0]
        X = normX[1:].reshape(self.matrix_shape)
        self.X = X
        U, D, V = self.SVD
        newD = np.zeros(D.shape[0]+1)
        newD[0] = norm
        newD[1:] = D
        newD = projl1_epigraph(newD) - newD
        result = np.zeros_like(normX)
        result[0] = newD[0]
        self.X = np.dot(U, newD[1:,np.newaxis] * V)
        result[1:] = self.X.reshape(-1)
        return result

#@objective_doc_templater()
class operator_norm_epigraph(svd_cone):
    
    def constraint(self, normX):
        """
        The non-negative constraint of x.
        """

        norm = normX[0]
        X = normX[1:].reshape(self.matrix_shape)
        self.X = X
        U, D, V = self.SVD

        incone = D.max() / norm <= 1 + self.tol
        if incone:
            return 0
        return np.inf


    def cone_prox(self, normX,  lipschitz=1):
        r"""
        Return (unique) minimizer

        .. math::

            v^{\lambda}(x) = \text{argmin}_{v \in \mathbb{R}^p} \frac{L}{2}
            \|x-v\|^2_2 \ \text{s.t.} \  v \in \mathbf{epi}(\ell_1)

        where *p*=x.shape[0], :math:`\lambda` = self.lagrange. 

        """
        norm = normX[0]
        X = normX[1:].reshape(self.matrix_shape)
        self.X = X
        U, D, V = self.SVD
        newD = np.zeros(D.shape[0]+1)
        newD[0] = norm
        newD[1:] = D
        newD = newD + projl1_epigraph(-newD)
        result = np.zeros_like(normX)
        result[0] = newD[0]
        self.X = np.dot(U, newD[1:,np.newaxis] * V)
        result[1:] = self.X.reshape(-1)
        return result

#@objective_doc_templater()
class operator_norm_epigraph_polar(svd_cone):
    
    def constraint(self, normX):
        """
        The non-negative constraint of x.
        """

        norm = normX[0]
        X = normX[1:].reshape(self.matrix_shape)
        self.X = X
        U, D, V = self.SVD

        incone = D.sum() / -norm <= 1 + self.tol
        if incone:
            return 0
        return np.inf


    def cone_prox(self, normX,  lipschitz=1):
        r"""
        Return (unique) minimizer

        .. math::

            v^{\lambda}(x) = \text{argmin}_{v \in \mathbb{R}^p} \frac{L}{2}
            \|x-v\|^2_2 \ \text{s.t.} \  v \in \mathbf{epi}(\ell_1)

        where *p*=x.shape[0], :math:`\lambda` = self.lagrange. 

        """
        norm = normX[0]
        X = normX[1:].reshape(self.matrix_shape)
        self.X = X
        U, D, V = self.SVD
        newD = np.zeros(D.shape[0]+1)
        newD[0] = norm
        newD[1:] = D
        newD = -projl1_epigraph(-newD)
        result = np.zeros_like(normX)
        result[0] = newD[0]
        self.X = np.dot(U, newD[1:,np.newaxis] * V)
        result[1:] = self.X.reshape(-1)
        return result



conjugate_svd_pairs = {}
conjugate_svd_pairs[nuclear_norm] = operator_norm
conjugate_svd_pairs[operator_norm] = nuclear_norm

conjugate_seminorm_pairs[nuclear_norm] = operator_norm
conjugate_seminorm_pairs[operator_norm] = nuclear_norm

conjugate_cone_pairs[nuclear_norm_epigraph] = nuclear_norm_epigraph_polar
conjugate_cone_pairs[nuclear_norm_epigraph_polar] = nuclear_norm_epigraph
conjugate_cone_pairs[operator_norm_epigraph] = operator_norm_epigraph_polar
conjugate_cone_pairs[operator_norm_epigraph_polar] = operator_norm_epigraph
