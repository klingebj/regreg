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

                
