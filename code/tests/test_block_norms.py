import numpy as np
import regreg.block_norms as B
import regreg.api as rr
import nose.tools as nt
import itertools

from test_seminorms import solveit, ac

@np.testing.dec.slow
def test_proximal_maps():
    shape = (5,4)

    bound = 0.14
    lagrange = 0.13


    Z = np.random.standard_normal(shape) * 2
    W = 0.02 * np.random.standard_normal(shape)
    U = 0.02 * np.random.standard_normal(shape)
    linq = rr.identity_quadratic(0,0,W,0)

    basis = np.linalg.svd(np.random.standard_normal((4,20)), full_matrices=0)[2]

    for L, atom, q, offset, FISTA, coef_stop in itertools.product([0.5,1,0.1], 
                                                       sorted(B.conjugate_block_pairs.keys()),
                                              [None, linq],
                                              [None, U],
                                              [False, True],
                                              [False, True]):

        if atom not in [B.block_max, B.block_sum]:
            p = atom(shape, quadratic=q, lagrange=lagrange,
                       offset=offset)
            d = p.conjugate 
            yield ac, p.lagrange_prox(Z, lipschitz=L), Z-d.bound_prox(Z*L, lipschitz=1./L)/L, 'testing lagrange_prox and bound_prox starting from atom %s ' % atom

            # some arguments of the constructor

            nt.assert_raises(AttributeError, setattr, p, 'bound', 4.)
            nt.assert_raises(AttributeError, setattr, d, 'lagrange', 4.)

            nt.assert_raises(AttributeError, setattr, p, 'bound', 4.)
            nt.assert_raises(AttributeError, setattr, d, 'lagrange', 4.)

            for t in solveit(p, Z, W, U, linq, L, FISTA, coef_stop):
                yield t

            b = atom(shape, bound=bound, quadratic=q,
                     offset=offset)

            for t in solveit(b, Z, W, U, linq, L, FISTA, coef_stop):
                yield t

