import numpy as np, sys
cimport numpy as np

"""
Implements prox and dual of group LASSO, strong set, seminorm and dual seminorm.
"""

DTYPE_float = np.float
ctypedef np.float_t DTYPE_float_t
DTYPE_int = np.int
ctypedef np.int_t DTYPE_int_t

#TODO: Add some documentation to this!

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

