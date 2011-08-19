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


def partial_svd(transform,
                r=1,
                extra_rank=2,
                max_its = 10000,
                tol = 1e-8,
                initial=None):

    """
    Compute the partial SVD of the linear_transform X using the Mazumder/Hastie algorithm in (TODO: CITE)
    """

    if isinstance(transform, np.ndarray):
        transform = linear_transform(transform)

    n = transform.dual_shape[0]
    p = transform.primal_shape[0]

    r = np.min([r,p])
    q = np.min([r + extra_rank, p])
    if initial is not None:
        if initial.shape == (n,q):
            U = initial
        elif initial.shape == (n,r):
            U = np.hstack([initial, np.random.standard_normal((n,q-r))])            
        else:
            raise ValueError("Initial value for U should have shape (%i,%i) or (%i,%i)" % (n,q,n,r))
    else:
        U = np.random.standard_normal((n,q))

    itercount = 0
    singular_rel_change = 1.
    old_singular_values = np.zeros(r)
    ind = range(r)
    while itercount < max_its and singular_rel_change > tol:
        print itercount, singular_rel_change
        VT,_ = np.linalg.qr(transform.adjoint_map(U))
        X_VT = transform.linear_map(VT)
        U,_ = np.linalg.qr(X_VT)
        singular_values = np.diagonal(np.dot(U.T,X_VT))[ind]
        singular_rel_change = np.linalg.norm(singular_values - old_singular_values)/np.linalg.norm(singular_values)
        old_singular_values = singular_values * 1.
        itercount += 1

    return U[:,ind] * np.sign(singular_values), np.fabs(singular_values),  VT[:,ind]


