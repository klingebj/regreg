import numpy as np, sys
cimport numpy as np

from .piecewise_linear import (find_solution_piecewise_linear,
                               find_solution_piecewise_linear_c)

"""
Implements (expected) linear time projections onto $\ell_1$ ball as described in
title = {Efficient projections onto the l1-ball for learning in high dimensions}
author = {Duchi, John and Shalev-Shwartz, Shai and Singer, Yoram and Chandra,
Tushar}
"""

DTYPE_float = np.float
ctypedef np.float_t DTYPE_float_t
DTYPE_int = np.int
ctypedef np.int_t DTYPE_int_t

def projl1(np.ndarray[DTYPE_float_t, ndim=1]  x, 
           DTYPE_float_t bound=1.):

    cdef np.ndarray[DTYPE_float_t, ndim=1] sorted_x = np.sort(np.fabs(x))
    cdef int p = x.shape[0]
    cdef double cut = find_solution_piecewise_linear_c(bound, 0, np.fabs(x))

    if cut < np.inf:
        return x
    else:
        return soft_threshold(x,cut)

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


def projl1_epigraph(np.ndarray[DTYPE_float_t, ndim=1] center):
    """
    Project center onto the l1 epigraph. The norm term is center[0],
    the coef term is center[1:]

    The l1 epigraph is the collection of points $(u,v): \|v\|_1 \leq u$
    np.fabs(coef).sum() <= bound.
    """

    cdef np.ndarray[DTYPE_float_t, ndim=1] x = center[1:]
    cdef np.ndarray[DTYPE_float_t, ndim=1] result = np.zeros_like(center)
    cdef DTYPE_float_t norm = center[0]
    cdef double cut = find_solution_piecewise_linear_c(norm, 1, np.fabs(x))
    cdef double cut2 = find_solution_piecewise_linear(norm, 1, np.fabs(x),
                                                      np.ones_like(x))

    if cut < np.inf:
        result[0] = norm + cut
        result[1:] = soft_threshold(x, cut)
    else:
        result = center
    return result


