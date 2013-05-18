import numpy as np
import regreg.atoms.cones as C
import regreg.api as rr
import nose.tools as nt
import itertools

from test_seminorms import Solver

@np.testing.dec.slow
def test_proximal_maps():
    shape = 20

    Z = np.random.standard_normal(shape) * 2
    W = 0.02 * np.random.standard_normal(shape)
    U = 0.02 * np.random.standard_normal(shape)
    quadratic = rr.identity_quadratic(0,0,W,0)

    for L, atom, q, offset, FISTA, coef_stop in itertools.product( 
        [0.5,1,0.1], 
        sorted(C.conjugate_cone_pairs.keys()),
        [None, quadratic],
        [None, U],
        [False, True],
        [False, True]):

        p = atom(shape, quadratic=q,
                   offset=offset)

        for t in Solver(p, Z, quadratic, L, FISTA, coef_stop).all():
            yield t
