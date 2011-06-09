import numpy as np
cimport numpy as np

"""
Implements (expected) linear time projections onto \ell_1 ball as described in
title = {Efficient projections onto the l1-ball for learning in high dimensions}
author = {Duchi, John and Shalev-Shwartz, Shai and Singer, Yoram and Chandra, Tushar}
"""

DTYPE_float = np.float
ctypedef np.float_t DTYPE_float_t
DTYPE_int = np.int
ctypedef np.int_t DTYPE_int_t

#TODO: Add some documentation to this!

def projl1(np.ndarray[DTYPE_float_t, ndim=1]  x, 
           DTYPE_float_t bound=1.):

    cdef int p = x.shape[0]
    cdef np.ndarray[DTYPE_int_t, ndim=2] U = np.empty((3,p),dtype=int)
    cdef int lenU = p
    cdef int Urow = 0
    cdef DTYPE_float_t s = 0
    cdef DTYPE_float_t rho = 0

    cdef int u, k, i, kind, Grow, Lrow, Gcol, Lcol, first
    cdef DTYPE_float_t xu, xk, ds, drho, eta

    first = 1
    while lenU:

        if Urow == 0:
            Lrow = 1
            Grow = 2
        elif Urow == 1:
            Lrow = 0
            Grow = 2
        else:
            Lrow = 0
            Grow = 1
            
        Lcol = 0
        Gcol = 0
        
        kind = np.random.randint(0,lenU)
        if first:
            k = kind
        else:
            k = U[Urow,kind]
        xk = x[k]
        if xk < 0:
            xk = -xk
        ds = 0
        drho = 0

        for i in range(lenU):
            if first:
                u = i
            else:
                u = U[Urow,i]
            xu = x[u]
            if xu < 0:
                xu = -xu
            if xu >= xk:
                if u == k:
                    ds += xu
                    drho += 1
                else:
                    U[Grow, Gcol] = u
                    Gcol += 1
                    ds += xu
                    drho += 1
            else:
                U[Lrow, Lcol] = u
                Lcol += 1

        if (s + ds) - (rho + drho)*xk < bound:
            s += ds
            rho += drho
            Urow = Lrow
            lenU = Lcol
        else:
            Urow = Grow
            lenU = Gcol
        first = 0
    eta = (s - bound)/rho
    if eta < 0:
        eta = 0.
    return soft_threshold(x, eta)
        

cdef soft_threshold(np.ndarray[DTYPE_float_t, ndim=1] x,
                    DTYPE_float_t lagrange):

    cdef int p = x.shape[0]
    cdef np.ndarray[DTYPE_float_t, ndim=1] y = np.empty(p)
    cdef DTYPE_float_t xi
    cdef int i
    for i in range(p):
        xi = x[i]
        if xi > 0:
            if xi < lagrange:
                y[i] = 0.
            else:
                y[i] = xi - lagrange
        else:
            if xi > -lagrange:
                y[i] = 0.
            else:
                y[i] = xi + lagrange
    return y

                
