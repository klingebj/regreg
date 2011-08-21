"""
This module contains the implementation operator and nuclear norms, used in 
matrix completion problems and other low-rank factorization
problems.

"""
import numpy as np
from atoms import atom, conjugate_seminorm_pairs
import warnings
from affine import linear_transform

try:
    from projl1_cython import projl1
except:
    warnings.warn('Cython version of projl1 not available. Using slower python version')
    from projl1_python import projl1


class factored_matrix(object):

    """
    A class for storing the SVD of a linear_tranform. Has affine_transform attributes like linear_map
    """

    def __init__(self,
                 linear_operator,
                 min_singular=0.,
                 tol=1e-5,
                 initial_rank=None):
        if min_singular >= 0:
            self.min_singular = min_singular
        else:
            raise ValueError("Minimum singular value must be non-negative")
        
        self.affine_offset = None
        self.tol = tol
        self.initial_rank = initial_rank
        if type(linear_operator) == type([]) and len(linear_operator) == 3:
            self.SVD = linear_operator
        else:
            self.X = linear_operator

    def _setX(self,transform):
        if isinstance(transform, np.ndarray):
            transform = linear_transform(transform)
        self.primal_shape = transform.primal_shape
        self.dual_shape = transform.dual_shape
        U, D, VT = compute_svd(transform, min_singular=self.min_singular, tol=self.tol, initial_rank = self.initial_rank)
        self.SVD = [U,D,VT]

    def _getX(self):
        return np.dot(self.SVD[0], np.dot(np.diag(self.SVD[1]), self.SVD[2]))
    X = property(_getX, _setX)

    def _getSVD(self):
        return self._SVD
    def _setSVD(self, SVD):
        if not (SVD[0].shape[1] == SVD[1].shape[0]) and (SVD[1].shape[2] == SVD[2].shape[0]):
            raise ValueError("SVD shapes are not compatible")
        self._SVD = SVD
    SVD = property(_getSVD, _setSVD)


    def linear_map(self, x):
        return np.dot(self.SVD[0], np.dot(np.diag(self.SVD[1]), np.dot(self.SVD[2], x)))

    def adjoint_map(self, x):
        return np.dot(self.SVD[2].T, np.dot(np.diag(self.SVD[1]), np.dot(self.SVD[0].T, x)))

    def affine_map(self,x):
        if self.affine_offset is None:
            return self.linear_map(x)
        else:
            return self.linear_map(x) + affine_offset

def compute_svd(transform,
                initial_rank = None,
                min_singular = 0.,
                tol = 1e-5):

    """
    Compute the SVD of a matrix
    """

    if isinstance(transform, np.ndarray):
        transform = linear_transform(transform)

    n = transform.dual_shape[0]
    p = transform.primal_shape[0]
    
    if initial_rank is None:
        r = np.round(np.min([n,p]) * 0.1)
    else:
        r = initial_rank

    min_so_far = 1.
    D = np.zeros(r)
    initial = None
    while len(D) >= r/2:
        U, D, VT = partial_svd(transform, r=r, extra_rank=5, tol=tol, initial=initial, return_full=True)
        if D[-1] < min_singular:
            break
        initial = U 
        r *= 2

    ind = np.where(D >= min_singular)[0]
    return U[:,ind], D[ind],  VT[ind,:]


def partial_svd(transform,
                r=1,
                extra_rank=2,
                max_its = 1000,
                tol = 1e-8,
                initial=None,
                return_full = False,
                debug=False):

    """
    Compute the partial SVD of the linear_transform X using the Mazumder/Hastie algorithm in (TODO: CITE)
    """

    if isinstance(transform, np.ndarray):
        transform = linear_transform(transform)

    n = transform.dual_shape[0]
    p = transform.primal_shape[0]


    r = np.int(np.min([r,p]))
    q = np.min([r + extra_rank, p])
    if initial is not None:
        if initial.shape == (n,q):
            U = initial
        else:
            U = np.hstack([initial, np.random.standard_normal((n,q-initial.shape[1]))])            
    else:
        U = np.random.standard_normal((n,q))

    if return_full:
        ind = np.arange(q)
        old_singular_values = np.zeros(q)
    else:
        ind = np.arange(r)
        old_singular_values = np.zeros(r)

    itercount = 0
    singular_rel_change = 1.


    while itercount < max_its and singular_rel_change > tol:
        if debug:
            print itercount, singular_rel_change
        V,_ = np.linalg.qr(transform.adjoint_map(U))
        X_V = transform.linear_map(V)
        U,_ = np.linalg.qr(X_V)
        singular_values = np.diagonal(np.dot(U.T,X_V))[ind]
        singular_rel_change = np.linalg.norm(singular_values - old_singular_values)/np.linalg.norm(singular_values)
        old_singular_values = singular_values * 1.
        itercount += 1

    nonzero = np.where(singular_values > 1e-12)[0]
    return U[:,ind[nonzero]] * np.sign(singular_values[nonzero]), np.fabs(singular_values[nonzero]),  V[:,ind[nonzero]].T


