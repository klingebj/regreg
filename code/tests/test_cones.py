import numpy as np
import regreg.atoms.cones as C
import regreg.atoms.svd_norms as C_SVD
import regreg.api as rr
import nose.tools as nt
import itertools

from test_seminorms import Solver

@np.testing.dec.slow
def test_proximal_maps():
    shape = 20

    W = 2 * np.random.standard_normal(shape)
    U = 0.02 * np.random.standard_normal(shape)
    quadratic = rr.identity_quadratic(0,0,W,0)

    for L, atom, q, offset, FISTA, coef_stop in itertools.product( 
        [0.5,1,0.1], 
        sorted(C.conjugate_cone_pairs.keys()),
        [None, quadratic],
        [True, False],
        [False, True],
        [False, True]):

        if atom in [C_SVD.nuclear_norm_epigraph,
                    C_SVD.nuclear_norm_epigraph_polar,
                    C_SVD.operator_norm_epigraph,
                    C_SVD.operator_norm_epigraph_polar]:
            shape = (20, 10)
            if offset:
                U = 0.02 * np.random.standard_normal(shape)
            else:
                U = None

            cone_instance = atom(shape, quadratic=q,
                                 offset=U)
            Z = np.random.standard_normal(cone_instance.shape)

            if U is not None:
                cone_instance.offset = 0.02 * np.random.standard_normal(cone_instance.shape)
            for t in Solver(cone_instance, Z, L, FISTA, coef_stop).all():
                yield t
