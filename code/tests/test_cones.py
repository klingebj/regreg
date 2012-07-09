import numpy as np
import regreg.cones as C
import regreg.api as rr
import nose.tools as nt
import itertools

from test_seminorms import solveit

@np.testing.dec.slow
def test_proximal_maps():
    shape = 20

    Z = np.random.standard_normal(shape) * 2
    W = 0.02 * np.random.standard_normal(shape)
    U = 0.02 * np.random.standard_normal(shape)
    linq = rr.identity_quadratic(0,0,W,0)

    for L, atom, q, offset, FISTA, coef_stop in itertools.product( 
        [0.5,1,0.1], 
        [C.l2_epigraph, C.l2_epigraph_polar],#sorted(C.conjugate_cone_pairs.keys()),
        [None, linq],
        [None, U],
        [False, True],
        [False, True]):

        p = atom(shape, quadratic=q,
                   offset=offset)

        for t in solveit(p, Z, W, U, linq, L, FISTA, coef_stop):
            yield t
