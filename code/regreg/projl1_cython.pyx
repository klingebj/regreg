import numpy as np, sys
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

    cdef np.ndarray[DTYPE_float_t, ndim=1] sorted_x = np.sort(np.fabs(x))
    cdef int p = x.shape[0]
    
    cdef double csum = 0.
    cdef double next, cut
    cdef int i, stop
    for i in range(p):
        next = sorted_x[p-i-1]
        csum += next
        stop = (csum - (i+1)*next) > bound
        if stop:
            break
    if stop:
        cut = next + (csum - (i+1)*next - bound)/(i)
        return soft_threshold(x,cut)
    else:
        return x

                                                            

def projl1_2(np.ndarray[DTYPE_float_t, ndim=1]  x, 
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

def projl1_epigraph(np.ndarray[DTYPE_float_t, ndim=1] center):
    """
    Project center onto the l1 epigraph. The norm term is center[0],
    the coef term is center[1:]

    The l1 epigraph is the collection of points (u,v): \|v\|_1 \leq u
    np.fabs(coef).sum() <= bound.

    """

    cdef np.ndarray[DTYPE_float_t, ndim=1] coef = center[1:]
    cdef DTYPE_float_t norm = center[0]
    cdef np.ndarray[DTYPE_float_t, ndim=1] sorted_coefs = np.sort(np.fabs(coef))

    cdef int n = sorted_coefs.shape[0]
    cdef np.ndarray[DTYPE_float_t, ndim=1] result = np.zeros(n+1, np.float)
    cdef int i, stop, idx
    cdef DTYPE_float_t csum = 0
    cdef DTYPE_float_t thold = sorted_coefs[n-1]
    cdef DTYPE_float_t x1, x2, y1, y2, slope
    
    # check to see if it's already in the epigraph

    if sorted_coefs.sum() <= norm:
        result[0] = norm
        result[1:] = coef
        return result
    x1 = sorted_coefs[n-1]
    y1 = - norm - x1
    for i in range(1, n-1):
        x2 = sorted_coefs[n-1-i]
        csum += x1
        y2 = (csum - i*x2) - (norm + x2)
        print x1, y1, x2, y2, np.fabs(soft_threshold(coef, x2)).sum() - norm - x2
        if y2 > 0:
            slope = (y1-y2) / (x1-x2)
            thold = (slope * x2 - y2) / slope
            print 'thold', thold
            break
        
        x1, y1 = x2, y2
    if thold != sorted_coefs[n-1]:
        result[0] = norm + thold
        result[1:] = soft_threshold(coef, thold)
    return result

def projlinf_epigraph(np.ndarray[DTYPE_float_t, ndim=1] center):
    """
    Project center onto the l-infinty epigraph. The norm term is center[0],
    the coef term is center[1:]

    The l-infinity epigraph is the collection of points (u,v): \|v\|_{\infty} \leq u
    np.fabs(coef).max() <= bound.

    """
    # we just use the fact that the polar of the linf epigraph is
    # is the negative of the l1 epigraph, so we project
    # -center onto the l1-epigraph and add the result to center...
    cdef np.ndarray[DTYPE_float_t, ndim=1] coef = -center[1:]
    cdef DTYPE_float_t norm = -center[0]
    cdef np.ndarray[DTYPE_float_t, ndim=1] sorted_coefs = np.sort(np.fabs(coef))

    cdef int n = sorted_coefs.shape[0]
    cdef np.ndarray[DTYPE_float_t, ndim=1] result = np.zeros(n+1, np.float)
    cdef int i, stop, idx
    cdef DTYPE_float_t csum = 0
    cdef DTYPE_float_t thold = sorted_coefs[n-1]
    cdef DTYPE_float_t x1, x2, y1, y2, slope
    
    # check to see if it's already in the epigraph

    if sorted_coefs.sum() <= norm:
        result[0] = norm
        result[1:] = coef
        return result
    x1 = sorted_coefs[n-1]
    y1 = - norm - x1
    for i in range(1, n-1):
        x2 = sorted_coefs[n-1-i]
        csum += x1
        y2 = (csum - i*x2) - (norm + x2)
        print x1, y1, x2, y2, np.fabs(soft_threshold(coef, x2)).sum() - norm - x2
        if y2 > 0:
            slope = (y1-y2) / (x1-x2)
            thold = (slope * x2 - y2) / slope
            print 'thold', thold
            break
        
        x1, y1 = x2, y2
    if thold != sorted_coefs[n-1]:
        result[0] = norm + thold
        result[1:] = soft_threshold(coef, thold)
    return center + result


def prox_group_lasso(np.ndarray[DTYPE_float_t, ndim=1] prox_center, 
                     DTYPE_float_t lagrange, DTYPE_float_t lipschitz,
                     np.ndarray[DTYPE_int_t, ndim=1] l1_penalty, 
                     np.ndarray[DTYPE_int_t, ndim=1] unpenalized,
                     np.ndarray[DTYPE_int_t, ndim=1] positive_part, 
                     np.ndarray[DTYPE_int_t, ndim=1] groups,
                     np.ndarray[DTYPE_float_t, ndim=1] weights):
    
    cdef np.ndarray norms = np.zeros_like(weights)
    cdef np.ndarray projection = np.zeros_like(prox_center)
    cdef int i
    cdef int p = groups.shape[0]
    
    cdef lf = lagrange / lipschitz
    
    for i in range(p):
        if groups[i] >= 0:
            norms[groups[i]] = norms[groups[i]] + prox_center[i]**2
    
    for j in range(weights.shape[0]):
        norms[j] = np.sqrt(norms[j])
        projection[groups == j] = prox_center[groups == j] / norms[j] * min(norms[j], lf * weights[j])
    
    projection[l1_penalty] = prox_center[l1_penalty] * np.minimum(1, lf / np.fabs(prox_center[l1_penalty]))
    projection[unpenalized] = 0
    projection[positive_part] = np.minimum(lf, prox_center[positive_part])
    
    return prox_center - projection

def project_group_lasso(np.ndarray[DTYPE_float_t, ndim=1] prox_center, 
                     DTYPE_float_t bound, 
                     np.ndarray[DTYPE_int_t, ndim=1] l1_penalty, 
                     np.ndarray[DTYPE_int_t, ndim=1] unpenalized,
                     np.ndarray[DTYPE_int_t, ndim=1] positive_part, 
                     np.ndarray[DTYPE_int_t, ndim=1] groups,
                     np.ndarray[DTYPE_float_t, ndim=1] weights):
    
    cdef np.ndarray norms = np.zeros_like(weights)
    cdef np.ndarray projection = np.zeros_like(prox_center)
    cdef int i
    cdef int p = groups.shape[0]
    
    for i in range(p):
        if groups[i] >= 0:
            norms[groups[i]] = norms[groups[i]] + prox_center[i]**2
    
    for j in range(weights.shape[0]):
        norms[j] = np.sqrt(norms[j])
        projection[groups == j] = prox_center[groups == j] / norms[j] * min(norms[j], bound * weights[j])
    
    projection[l1_penalty] = prox_center[l1_penalty] * np.minimum(1, bound / np.fabs(prox_center[l1_penalty]))
    projection[unpenalized] = 0
    projection[positive_part] = np.minimum(bound, prox_center[positive_part])
    
    return projection

def seminorm_group_lasso(np.ndarray[DTYPE_float_t, ndim=1] x, 
                         np.ndarray[DTYPE_int_t, ndim=1] l1_penalty, 
                         np.ndarray[DTYPE_int_t, ndim=1] unpenalized,
                         np.ndarray[DTYPE_int_t, ndim=1] positive_part, 
                         np.ndarray[DTYPE_int_t, ndim=1] groups,
                         np.ndarray[DTYPE_float_t, ndim=1] weights,
                         DTYPE_int_t check_feasibility):
    
    cdef np.ndarray norms = np.zeros_like(weights)
    cdef int i
    cdef DTYPE_float_t value
    cdef int p = groups.shape[0]
    
    for i in range(p):
        if groups[i] >= 0:
            norms[groups[i]] = norms[groups[i]] + x[i]**2
    
    value = np.fabs(x[l1_penalty]).sum()
    value += np.maximum(x[positive_part], 0).sum()

    for j in range(weights.shape[0]):
        norms[j] = np.sqrt(norms[j])
        value += weights[j] * norms[j]

    tol = 1.e-5
    if check_feasibility:
        xpos = x[positive_part]
        if tuple(xpos.shape) not in [(),(0,)] and xpos.min() < tol:
            value = np.inf
    return value


def strong_set_group_lasso(np.ndarray[DTYPE_float_t, ndim=1] x, 
                           DTYPE_float_t lagrange_new,
                           DTYPE_float_t lagrange_cur,
                           DTYPE_float_t slope_estimate,
                           np.ndarray[DTYPE_int_t, ndim=1] l1_penalty, 
                           np.ndarray[DTYPE_int_t, ndim=1] unpenalized,
                           np.ndarray[DTYPE_int_t, ndim=1] positive_part, 
                           np.ndarray[DTYPE_int_t, ndim=1] groups,
                           np.ndarray[DTYPE_float_t, ndim=1] weights):
    
    cdef np.ndarray value = np.zeros_like(x)
    cdef np.ndarray norms = np.zeros_like(weights)
    cdef int i
    cdef int p = groups.shape[0]
    
    for i in range(p):
        if groups[i] >= 0:
            norms[groups[i]] = norms[groups[i]] + x[i]**2
    
    value[l1_penalty] = np.fabs(x[l1_penalty]) < (slope_estimate+1)*lagrange_new - slope_estimate*lagrange_cur
    value[positive_part] = -x[positive_part] < (slope_estimate+1) * lagrange_new - slope_estimate*lagrange_cur

    for j in range(weights.shape[0]):
        norms[j] = np.sqrt(norms[j])
        value[groups == j] = norms[j] < (slope_estimate+1) * lagrange_new - slope_estimate*lagrange_cur

    return 1 - value

    
def seminorm_group_lasso_conjugate(np.ndarray[DTYPE_float_t, ndim=1] x, 
                                   np.ndarray[DTYPE_int_t, ndim=1] l1_penalty, 
                                   np.ndarray[DTYPE_int_t, ndim=1] unpenalized,
                                   np.ndarray[DTYPE_int_t, ndim=1] positive_part, 
                                   np.ndarray[DTYPE_int_t, ndim=1] groups,
                                   np.ndarray[DTYPE_float_t, ndim=1] weights):
    
    cdef np.ndarray norms = np.zeros_like(weights)
    cdef int i
    cdef DTYPE_float_t value
    cdef int p = groups.shape[0]
    
    for i in range(p):
        if groups[i] >= 0:
            norms[groups[i]] = norms[groups[i]] + x[i]**2
    
    xl1 = x[l1_penalty]
    if xl1.shape not in [(), (0,)]:
        value = np.fabs(xl1).max()
    else:
        value = -np.inf

    xpos = x[positive_part]
    if xpos.shape not in [(), (0,)]:
        value = max(value, np.maximum(xpos, 0).max())

    for j in range(weights.shape[0]):
        norms[j] = np.sqrt(norms[j])
        value = max(value, weights[j] * norms[j])

    return value

    
