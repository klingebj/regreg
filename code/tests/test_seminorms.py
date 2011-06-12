import numpy as np
import regreg.atoms as A
import nose.tools as nt

def ac(x,y):
    return nt.assert_true(np.linalg.norm(x-y) <= 1.0e-04 * max([1, np.linalg.norm(x), np.linalg.norm(y)]))

def test_proximal_maps():
    bound = 0.14
    lagrange = 0.5
    shape = 50

    Z = np.random.standard_normal(shape)
    for primal, dual in A.primal_dual_seminorm_pairs.items():
        p = primal(shape, lagrange=lagrange)
        d = p.conjugate
        print p, d
        yield nt.assert_equal, d, dual(shape, bound=lagrange)
        yield ac, Z-p.proximal(Z), d.proximal(Z)
        #yield ac, d.proximal_optimum(Z)[1] + np.linalg.norm(Z)**2/2, p.proximal_optimum(Z)[1]

        d = dual(shape, bound=bound)
        p = d.conjugate
        yield nt.assert_equal, p, primal(shape, lagrange=bound)
        yield ac, Z-p.proximal(Z), d.proximal(Z)
#        yield ac, d.proximal_optimum(Z)[1], p.proximal_optimum(Z)[1] + np.linalg.norm(Z)**2/2

def test_linear_term_proximal():
    bound = 0.14
    lagrange = 0.5
    shape = 50

    Z = np.random.standard_normal(shape)
    W = 0.02 * np.random.standard_normal(shape)
    for primal, dual in A.primal_dual_seminorm_pairs.items():
        p = primal(shape, lagrange=lagrange, linear_term=W)
        d = p.conjugate
        print p, d
        yield nt.assert_equal, d, dual(shape, bound=lagrange)
        yield ac, Z-p.proximal(Z), d.proximal(Z)
        #yield ac, d.proximal_optimum(Z)[1] + np.linalg.norm(Z)**2/2, p.proximal_optimum(Z)[1]

        d = dual(shape, bound=bound, linear_term=W)
        p = d.conjugate
        yield nt.assert_equal, p, primal(shape, lagrange=bound)
        yield ac, Z-p.proximal(Z), d.proximal(Z)
#        yield ac, d.proximal_optimum(Z)[1], p.proximal_optimum(Z)[1] + np.linalg.norm(Z)**2/2

def test_offset_proximal():
    bound = 0.14
    lagrange = 0.5
    shape = 50

    Z = np.random.standard_normal(shape)
    W = 0.02 * np.random.standard_normal(shape)
    for primal, dual in A.primal_dual_seminorm_pairs.items():
        p = primal(shape, lagrange=lagrange, offset=W)
        d = p.conjugate
        print p, d
        yield nt.assert_equal, d, dual(shape, bound=lagrange)
        yield ac, Z-p.proximal(Z), d.proximal(Z)
        #yield ac, d.proximal_optimum(Z)[1] + np.linalg.norm(Z)**2/2, p.proximal_optimum(Z)[1]

        d = dual(shape, bound=bound, offset=W)
        p = d.conjugate
        yield nt.assert_equal, p, primal(shape, lagrange=bound)
        yield ac, Z-p.proximal(Z), d.proximal(Z)
#        yield ac, d.proximal_optimum(Z)[1], p.proximal_optimum(Z)[1] + np.linalg.norm(Z)**2/2

def test_offset_and_linear_term_proximal():
    bound = 0.14
    lagrange = 0.5
    shape = 50

    Z = np.random.standard_normal(shape)
    W = 0.02 * np.random.standard_normal(shape)
    U = 0.02 * np.random.standard_normal(shape)
    for primal, dual in A.primal_dual_seminorm_pairs.items():
        p = primal(shape, lagrange=lagrange, offset=W, linear_term=U)
        d = p.conjugate
        print p, d
        yield nt.assert_equal, d, dual(shape, bound=lagrange)
        yield ac, Z-p.proximal(Z), d.proximal(Z)
        #yield ac, d.proximal_optimum(Z)[1] + np.linalg.norm(Z)**2/2, p.proximal_optimum(Z)[1]

        d = dual(shape, bound=bound, offset=W, linear_term=U)
        p = d.conjugate
        yield nt.assert_equal, p, primal(shape, lagrange=bound)
        yield ac, Z-p.proximal(Z), d.proximal(Z)
#        yield ac, d.proximal_optimum(Z)[1], p.proximal_optimum(Z)[1] + np.linalg.norm(Z)**2/2
