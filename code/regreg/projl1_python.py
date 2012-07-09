import numpy as np

"""
Implements (expected) linear time projections onto \ell_1 ball as described in
title = {Efficient projections onto the l1-ball for learning in high dimensions}
author = {Duchi, John and Shalev-Shwartz, Shai and Singer, Yoram and Chandra, Tushar}
"""

#A faster Cython implementation is in projl1_cython.pyx


def projl1(x, 
           bound=1.):

    sorted_x = np.sort(np.fabs(x))
    p = x.shape[0]
    
    csum = 0.
    for i in range(p):
        next = sorted_x[p-i-1]
        csum += next
        stop = (csum - (i+1)*next) > bound
        if stop:
            break
    if stop:
        cut = next + (csum - (i+1)*next - bound)/(i)
        return np.sign(x) * np.maximum(np.fabs(x)-cut,0.)
    else:
        return x

                
def projl1_epigraph(center):
    """
    Project center=proxq.true_center onto the l1 epigraph. The bound term is center[0],
    the coef term is center[1:]

    The l1 epigraph is the collection of points (u,v): \|v\|_1 \leq u
    np.fabs(coef).sum() <= bound.

    """
    
    norm = center[0]
    coef = center[1:]
    sorted_coefs = np.sort(np.fabs(coef))
    n = sorted_coefs.shape[0]
    
    csum = sorted_coefs.sum()
    for i, c in enumerate(sorted_coefs):
        csum -= c
        if csum - (n - i - 1) * c <= norm + c:
            # this will terminate as long as norm >= 0
            # when it terminates, we know that the solution is between
            # sorted_coefs[i-1] and sorted_coefs[i]
            
            # we set the cumulative sum back to the value at i-1
            csum += c
            idx = i-1
            break
    if i == n-1: # if it hasn't terminated early, then even soft-thresholding at the largest value was insufficent, answer is 0
        return np.zeros_like(center)
    
    # the solution is such that csum - (n-idx-1)*x = norm+x
    thold = (csum - norm) / (n-idx)
    result = np.zeros_like(center)
    result[0] = norm + thold
    result[1:] = st(coef, thold) 
    return result
