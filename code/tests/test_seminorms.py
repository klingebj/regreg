import numpy as np
import regreg.atoms as A
import nose.tools as nt

def test_proximal_maps():
    bound = 0.14
    lagrange = 0.5
    shape = 50

    def ac(x,y):
        return np.testing.assert_allclose(x, y, rtol=1.0e-03)
    Z = np.random.standard_normal(shape)
    for primal, dual in A.primal_dual_seminorm_pairs.items():
        p = primal(shape, lagrange=lagrange)
        d = p.conjugate
        yield nt.assert_equal, d, dual(shape, bound=lagrange)
        yield ac, Z-p.proximal(Z), d.proximal(Z)

        d = dual(shape, bound=bound)
        p = d.conjugate
        yield nt.assert_equal, p, primal(shape, lagrange=bound)
        yield ac, Z-p.proximal(Z), d.proximal(Z)
