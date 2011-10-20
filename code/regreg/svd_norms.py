"""
This module contains the implementation operator and nuclear norms, used in 
matrix completion problems and other low-rank factorization
problems.

"""
from md5 import md5
import numpy as np
from projl1 import projl1
from atoms import atom, conjugate_seminorm_pairs
from copy import copy

class svd_atom(atom):
    
    _doc_dict = {'linear':r' + \text{Tr}(\eta^T X)',
                 'constant':r' + \tau',
                 'objective': '',
                 'shape':r'p \times q',
                 'var':r'X'}
    
    def compute_and_store_svd(self, X):
        """
        Compute and store svd of X for use in multiple function calls.
        """
        self._md5x = md5(X).hexdigest()
        self._X = X
        self.SVD = np.linalg.svd(X)
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
        atom.get_conjugate(self)
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

    def lagrange_prox(self, X, lipschitz=1, lagrange=None):
        r"""
        Return unique minimizer

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

        If the argument lagragne is None and the atom is in lagrange mode, 
        self.lagrange is used as the lagrange parameter, 
        else an exception is raised.

        The class atom's lagrange_prox just returns the appropriate lagrange
        parameter for use by the subclasses.
        """
        if lagrange is None:
            lagrange = self.lagrange
        if lagrange is None:
            raise ValueError('either atom must be in Lagrange mode or a keyword "lagrange" argument must be supplied')
        return lagrange

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

        If the argument is bound None and the atom is in bound mode, 
        self.bound is used as the bound parameter, 
        else an exception is raised.

        The class atom's bound_prox just returns the appropriate bound
        parameter for use by the subclasses.

        """
        if bound is None:
            bound = self.bound
        if bound is None:
            raise ValueError('either atom must be in bound mode or a keyword "bound" argument must be supplied')
        return bound

class nuclear_norm(svd_atom):

    """
    The nuclear norm
    """
    prox_tol = 1.0e-10

    objective_template = r"""\|%(var)s\|_*"""
    _doc_dict = copy(svd_atom._doc_dict)
    _doc_dict['objective'] = objective_template % {'var': r'X + A'}

    def seminorm(self, X, check_feasibility=False,
                 lagrange=None):
        # This will compute an svd of X
        # if the md5 hash of X doesn't match.
        lagrange = atom.seminorm(self, X, lagrange=lagrange,
                                 check_feasibility=check_feasibility)
        self.X = X
        _, D, _ = self.SVD
        return self.lagrange * np.sum(D)
    seminorm.__doc__ = atom.seminorm.__doc__ % _doc_dict

    def constraint(self, X, bound=None):
        # This will compute an svd of X
        # if the md5 hash of X doesn't match.
        bound = atom.constraint(self, X, bound=bound)
        self.X = X
        _, D, _ = self.SVD
        inbox = np.sum(D) <= bound * (1 + self.tol)
        if inbox:
            return 0
        else:
            return np.inf
    constraint.__doc__ = atom.constraint.__doc__ % _doc_dict

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
    lagrange_prox.__doc__ = svd_atom.lagrange_prox.__doc__ % _doc_dict

    def bound_prox(self, X, lipschitz=1, bound=None):
        bound = svd_atom.bound_prox(self, X, lipschitz, bound)
        self.X = X
        U, D, V = self.SVD
        D_projected = projl1(D, self.bound)
        keepD = D_projected > 0
        # store the projected X -- or should we keep original?
        self.SVD = U[:,keepD], D_projected[keepD], V[keepD]
        c = self.conjugate
        c.SVD = self.SVD
        self._X = np.dot(U[:,keepD], D_projected[keepD][:,np.newaxis] * V[keepD])
        return self.X

    bound_prox.__doc__ = svd_atom.bound_prox.__doc__ % _doc_dict

class operator_norm(svd_atom):

    """
    The operator norm
    """
    prox_tol = 1.0e-10

    objective_template = r"""\|%(var)s\|_{\text{op}}"""
    _doc_dict = copy(svd_atom._doc_dict)
    _doc_dict['objective'] = objective_template % {'var': r'X + A'}

    def seminorm(self, X, lagrange=None, check_feasibility=False):
        # This will compute an svd of X
        # if the md5 hash of X doesn't match.
        lagrange = atom.seminorm(self, X, lagrange=lagrange,
                                 check_feasibility=check_feasibility)
        self.X = X
        _, D, _ = self.SVD
        return self.lagrange * np.max(D)
    seminorm.__doc__ = atom.seminorm.__doc__ % _doc_dict

    def constraint(self, X, bound=None):
        # This will compute an svd of X
        # if the md5 hash of X doesn't match.
        bound = atom.constraint(self, X, bound=bound)
        self.X = X
        _, D, _ = self.SVD
        inbox = np.max(D) <= self.bound * (1 + self.tol)
        if inbox:
            return 0
        else:
            return np.inf
    constraint.__doc__ = atom.constraint.__doc__ % _doc_dict

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
    lagrange_prox.__doc__ = svd_atom.lagrange_prox.__doc__ % _doc_dict

    def bound_prox(self, X, lipschitz=1, bound=None):
        bound = svd_atom.bound_prox(self, X, lipschitz, bound)
        self.X = X
        self._D = U, np.maximum(D, self.bound), V
        U, D, V = self.SVD
        # store the projected X -- or should we keep original?
        self._X = np.dot(U, D[:,np.newaxis] * V)
        return self.X
    bound_prox.__doc__ = svd_atom.bound_prox.__doc__ % _doc_dict

conjugate_seminorm_pairs[nuclear_norm] = operator_norm
conjugate_seminorm_pairs[operator_norm] = nuclear_norm
