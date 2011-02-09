# cython: profile=True

import numpy as np
cimport numpy as np
import time

## Local imports

import subfunctions as sf

## Compile-time datatypes
DTYPE_float = np.float
ctypedef np.float_t DTYPE_float_t
DTYPE_int = np.int
ctypedef np.int_t DTYPE_int_t

def _update_lasso_cwpath(np.ndarray[DTYPE_int_t, ndim=1] active,
                         penalty,
                         list nonzero,
                         np.ndarray[DTYPE_float_t, ndim=1] beta,
                         np.ndarray[DTYPE_float_t, ndim=1] r,
                         _X,
                         np.ndarray[DTYPE_float_t, ndim=1] ssq,
                         DTYPE_int_t inner_its,
                         DTYPE_int_t update_nonzero = False,
                         DTYPE_float_t tol = 1e-3):

                
    """
    Do coordinate-wise update of lasso coefficients beta
    in active set.

    The coordinates that are nonzero are stored in the
    list nonzero.

    Optimizes the LASSO penalty

    norm(Y-dot(X,b))**2/2 + penalty*fabs(b).sum()

    as a function of b.

    """
    cdef list X
    cdef DTYPE_float_t S, new, db, l1
    cdef DTYPE_int_t q, n, i, j, k, m, count, stop
    cdef np.ndarray[DTYPE_float_t, ndim=1] bold

    X = sf.aslist(_X)
    l1 = float(penalty['l1'])
    q = active.shape[0]
    n = r.shape[0]
    cdef np.ndarray[DTYPE_float_t, ndim=1] col = np.empty(n)

    count = 0
    stop = False 
    while count < inner_its and not stop:
        bold = beta.copy()
        for j in range(q):
            i = active[j]
            
            #Select appropriate column
            #for k in range(n):
            #    col[k] = X[k][i]
            col = sf.select_col(X,n,i)
                            
            S = beta[i] * ssq[i]
            S += np.dot(col,r)
            new = _solve_plin(ssq[i]/2,
                              -S,
                              l1)
            if update_nonzero:
                if new != 0:
                    nonzero.append(i)
            db = beta[i] - new
            r += (db * col)
            beta[i] = new
        stop = sf.coefficientCheck(bold,beta,tol)
        count += 1

def _update_lasso_wts(np.ndarray[DTYPE_int_t, ndim=1] active,
                      penalty,
                      list nonzero,
                      np.ndarray[DTYPE_float_t, ndim=1] beta,
                      np.ndarray[DTYPE_float_t, ndim=1] r,
                      list X,
                      np.ndarray[DTYPE_float_t, ndim=1] ssq,
                      np.ndarray[DTYPE_float_t, ndim=1] wts,
                      DTYPE_int_t inner_its,
                      DTYPE_int_t update_nonzero = False,
                      DTYPE_float_t tol = 1e-3):
                
    """
    Do coordinate-wise update of lasso coefficients beta
    in active set.

    The coordinates that are nonzero are stored in the
    list nonzero.

    Optimizes the LASSO penalty

    norm(Y-dot(X,b))**2/2 + penalty*fabs(b).sum()

    as a function of b.

    """
    cdef DTYPE_float_t S, new, db, l1
    cdef DTYPE_int_t q, n, i, j, k, m, count, stop
    cdef np.ndarray[DTYPE_float_t, ndim=1] bold


    l1 = float(penalty['l1'])
    q = active.shape[0]
    n = r.shape[0]
    cdef np.ndarray[DTYPE_float_t, ndim=1] col = np.empty(n)

    count = 0
    stop = False 
    while count < inner_its and not stop:
        bold = beta.copy()
        for j in range(q):
            i = active[j]
            
            #Select appropriate column
            for k in range(n):
                col[k] = X[k][i] * wts[k]
            
                            
            S = beta[i] * ssq[i] 
            S += np.dot(col,r)
            new = _solve_plin(ssq[i]/(2*n),
                              -(S/n),
                              l1)
            if update_nonzero:
                if new != 0:
                    nonzero.append(i)
            db = beta[i] - new
            r += (db * col)
            beta[i] = new
        stop = sf.coefficientCheck(bold,beta,tol)
        count += 1
    #print count

def _update_graphnet_cwpath(np.ndarray[DTYPE_int_t, ndim=1] active,
                            penalty,
                            list nonzero,
                            np.ndarray[DTYPE_float_t, ndim=1] beta,
                            np.ndarray[DTYPE_float_t, ndim=1] r,
                            list X,
                            np.ndarray[DTYPE_float_t, ndim=1] ssq,
                            np.ndarray[DTYPE_int_t, ndim=2] adj,
                            np.ndarray[DTYPE_int_t, ndim=1] nadj,
                            DTYPE_int_t inner_its,
                            DTYPE_int_t update_nonzero = False,
                            DTYPE_float_t tol = 1e-3):

    """
    Do coordinate-wise update of lasso coefficients beta
    in active set.

    The coordinates that are nonzero are stored in the
    list nonzero.

    """
    cdef double S, lin, quad, new, db, l1, l2, l3
    cdef long q, n, i, j, k, count, stop
    cdef np.ndarray[DTYPE_float_t, ndim=1] bold


    l1 = float(penalty['l1'])
    l2 = float(penalty['l2'])
    l3 = float(penalty['l3'])
    q = active.shape[0]
    n = r.shape[0]
    cdef np.ndarray[DTYPE_float_t, ndim=1] col = np.empty(n)

    count = 0
    stop = False
    while count < inner_its and not stop:
        bold = beta.copy()
        for j in range(q):
            i = active[j]

            #Select appropriate column
            for k in range(n):
                col[k] = X[k][i]

                
            S = beta[i] * ssq[i]
            S += np.dot(col,r)

            lin, quad = sf._compute_Lbeta(adj,nadj,beta,i)
            new = _solve_plin(ssq[i]/(2*n) + l3*quad/2. + l2/2.,
                              -(S/n)+l3*lin/2., 
                              l1)
            if update_nonzero:
                if new != 0:
                    nonzero.append(i)
            db = beta[i] - new
            r += (db * col)
            beta[i] = new
        stop = sf.coefficientCheck(bold,beta,tol)
        count += 1


def _update_lin_graphnet_cwpath(np.ndarray[DTYPE_int_t, ndim=1] active,
                                penalty,
                                list nonzero,
                                np.ndarray[DTYPE_float_t, ndim=1] beta,
                                np.ndarray[DTYPE_float_t, ndim=1] Y,
                                list X,
                                np.ndarray[DTYPE_float_t, ndim=1] inner,
                                np.ndarray[DTYPE_int_t, ndim=2] adj,
                                np.ndarray[DTYPE_int_t, ndim=1] nadj,
                                DTYPE_int_t inner_its,
                                DTYPE_int_t update_nonzero = False,
                                DTYPE_float_t tol = 1e-3):

    """
    Do coordinate-wise update of lasso coefficients beta
    in active set.

    The coordinates that are nonzero are stored in the
    list nonzero.

    """
    cdef double S, lin, quad, new, db, l1, l2, l3
    cdef long q, n, i, j, k, count, stop
    cdef np.ndarray[DTYPE_float_t, ndim=1] bold


    l1 = float(penalty['l1'])
    l2 = float(penalty['l2'])
    l3 = float(penalty['l3'])
    q = active.shape[0]
    n = len(Y)

    count = 0
    stop = False
    while count < inner_its and not stop:
        bold = beta.copy()
        for j in range(q):
            i = active[j]
            S = inner[i]
            if l3 > 0.:
                lin, quad = sf._compute_Lbeta(adj,nadj,beta,i)
            else:
                lin = 0.
                quad = 0.
            new = _solve_plin(l3*quad/2. + l2/2.,
                              -S+l3*lin/2., 
                              l1)
            if update_nonzero:
                if new != 0:
                    nonzero.append(i)
            db = beta[i] - new
            beta[i] = new
        stop = sf.coefficientCheck(bold,beta,tol)
        count += 1

def _update_graphnet_wts(np.ndarray[DTYPE_int_t, ndim=1] active,
                         penalty,
                         list nonzero,
                         np.ndarray[DTYPE_float_t, ndim=1] beta,
                         np.ndarray[DTYPE_float_t, ndim=1] r,
                         list X,
                         np.ndarray[DTYPE_float_t, ndim=1] ssq,
                         np.ndarray[DTYPE_int_t, ndim=2] adj,
                         np.ndarray[DTYPE_int_t, ndim=1] nadj,
                         np.ndarray[DTYPE_float_t, ndim=1] wts,
                         DTYPE_int_t inner_its,
                         DTYPE_int_t update_nonzero = False,
                         DTYPE_float_t tol = 1e-3):

    """
    Do coordinate-wise update of lasso coefficients beta
    in active set.

    The coordinates that are nonzero are stored in the
    list nonzero.

    """
    cdef double S, lin, quad, new, db, l1, l2, l3
    cdef long q, n, i, j, k, count, stop
    cdef np.ndarray[DTYPE_float_t, ndim=1] bold


    l1 = float(penalty['l1'])
    l2 = float(penalty['l2'])
    l3 = float(penalty['l3'])
    q = active.shape[0]
    n = r.shape[0]
    cdef np.ndarray[DTYPE_float_t, ndim=1] col = np.empty(n)

    count = 0
    stop = False
    while count < inner_its and not stop:
        bold = beta.copy()
        for j in range(q):
            i = active[j]

            #Select appropriate column
            for k in range(n):
                col[k] = X[k][i] * wts[k]

                
            S = beta[i] * ssq[i]
            S += np.dot(col,r)

            lin, quad = sf._compute_Lbeta(adj,nadj,beta,i)
            new = _solve_plin(ssq[i]/(2*n) + l3*quad/2. + l2/2.,
                              -(S/n)+l3*lin/2., 
                              l1)
            if update_nonzero:
                if new != 0:
                    nonzero.append(i)
            db = beta[i] - new
            r += (db * col)
            beta[i] = new
        stop = sf.coefficientCheck(bold,beta,tol)
        count += 1
                   
cdef DTYPE_float_t _solve_plin(DTYPE_float_t a,
                               DTYPE_float_t b,
                               DTYPE_float_t c):
    """
    Find the minimizer of

    a*x**2 + b*x + c*fabs(x)

    for positive constants a, c and arbitrary b.
    """

    if b < 0:
        if b > -c:
            return 0.
        else:
            return -(c + b) / (2.*a)
    else:
        if c > b:
            return 0.
        else:
            return (c - b) / (2.*a)
