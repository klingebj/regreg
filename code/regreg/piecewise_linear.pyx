import numpy as np, sys
cimport numpy as np


DTYPE_float = np.float
ctypedef np.float_t DTYPE_float_t
DTYPE_int = np.int
ctypedef np.int_t DTYPE_int_t


def find_solution_piecewise_linear(DTYPE_float_t b,
                                   DTYPE_float_t slope,
                                   np.ndarray[DTYPE_float_t, ndim=1] norms,
                                   np.ndarray[DTYPE_float_t, ndim=1] weights):
    """
    Given a piecewise linear function of the form

       f(t) = (weights[i] * (weights[i] * t < norms[i]) * (norms[i] - weights[i] * t)).sum()

    with weights[i] >= 0

    Return the t>=0 such that f(t)=slope*t+b, if one exists. 
    Else, it returns np.inf.

    That is, it returns

    inf (t >= 0: f(t) >= slope*t + b)

    This function is used in projecting onto 
    l1 balls of size s and epigraphs. The ball projection uses slope=0,
    b=s,
    and if this returns np.inf, one is already within the ball.

    The epigraph projection uses slope=1 and b=the norm
    part of the epigraph. If this
    returns np.inf, one is already within the epigraph.

    """

    cdef int q = norms.shape[0]
    cdef int stop = 0

    if (weights * norms).sum() < b:
        return np.inf

    cdef np.ndarray[DTYPE_float_t, ndim=1] knots = norms / weights
    cdef np.ndarray[DTYPE_int_t, ndim=1] order = np.argsort(knots)

    slope = - slope # move the linear piece to the LHS
    cdef double curX = knots[order[q-1]]
    cdef double curV = slope * curX
    
    cdef double nextX, nextV
    cdef double solution = np.inf
    
    # if f(0) < b, then the set is empty
    # \inf of empty set is +\infty
    
    for j in range(q-1):
        slope -= weights[order[q-j-1]]**2
        nextX = knots[order[q-j-2]]
        nextV = curV + slope * (nextX - curX)
        stop = nextV > b
        if stop:
            intercept = curV - curX * slope
            break
        curV, curX = nextV, nextX
        curX = nextX
         
    if stop:
        solution = (b - intercept) / slope
    else:
        slope -= weights[order[0]]**2
        nextX = 0
        nextV = curV + slope * (nextX - curX)
        intercept = curV - curX * slope
        solution = (b - intercept) / slope
    return solution

def find_solution_piecewise_linear_c(DTYPE_float_t b,
                                   DTYPE_float_t slope,
                                   np.ndarray[DTYPE_float_t, ndim=1] norms):
    """
    Given a piecewise linear function of the form

       f(t) = ((t < norms[i]) * (norms[i] - t)).sum()

    Return the t>=0 such that f(t)=slope*t+b, if one exists. 
    Else, it returns np.inf.

    That is, it returns

    inf (t >= 0: f(t) >= slope*t + b)

    This function is used in projecting onto 
    l1 balls of size s and epigraphs. The ball projection uses slope=0,
    b=s,
    and if this returns np.inf, one is already within the ball.

    The epigraph projection uses slope=1 and b=the norm
    part of the epigraph. If this
    returns np.inf, one is already within the epigraph.

    """

    cdef int q = norms.shape[0]
    cdef int stop = 0

    if (norms).sum() < b:
        return np.inf

    cdef np.ndarray[DTYPE_float_t, ndim=1] knots = norms
    cdef np.ndarray[DTYPE_int_t, ndim=1] order = np.argsort(knots)

    slope = - slope # move the linear piece to the LHS
    cdef double curX = knots[order[q-1]]
    cdef double curV = slope * curX
    
    cdef double nextX, nextV
    cdef double solution = np.inf
    
    # if f(0) < b, then the set is empty
    # \inf of empty set is +\infty
    
    for j in range(q-1):
        slope -= 1.
        nextX = knots[order[q-j-2]]
        nextV = curV + slope * (nextX - curX)
        stop = nextV > b
        if stop:
            intercept = curV - curX * slope
            break
        curV, curX = nextV, nextX
        curX = nextX
         
    if stop:
        solution = (b - intercept) / slope
    else:
        slope -= 1.
        nextX = 0
        nextV = curV + slope * (nextX - curX)
        intercept = curV - curX * slope
        solution = (b - intercept) / slope
    return solution

