import numpy as np
import regreg.cones as C
import regreg.api as rr
import nose.tools as nt

def ac(x,y):
    print x, y
    return nt.assert_true(np.linalg.norm(x-y) <= 1.0e-04 * max([1, np.linalg.norm(x), np.linalg.norm(y)]))

def test_proximal_maps():
    shape = 20

    Z = np.random.standard_normal(shape)
    for primal, dual in C.conjugate_cone_pairs.items():
        p = primal(shape)
        d = p.conjugate
        yield nt.assert_equal, d, dual(shape)
        yield ac, Z-p.proximal(Z), d.proximal(Z)

        loss = rr.quadratic.shift(-Z, coef=0.5)
        problem = rr.composite(loss.smooth_objective, p.nonsmooth_objective,
                               p.proximal, np.random.standard_normal(shape))
        solver = rr.FISTA(problem)
        solver.fit(tol=1.0e-12)

        yield ac, p.proximal(Z), solver.composite.coefs

        loss = rr.quadratic.shift(-Z, coef=0.5)
        problem = rr.container(loss, p)
        solver = rr.FISTA(problem)
        solver.fit(tol=1.0e-12)

        yield ac, p.proximal(Z), solver.composite.coefs

        problem = rr.composite(loss.smooth_objective, d.nonsmooth_objective,
                               d.proximal, np.random.standard_normal(shape))
        solver = rr.FISTA(problem)
        solver.fit(tol=1.0e-12)
        yield ac, d.proximal(Z), solver.composite.coefs

        #yield ac, d.proximal_optimum(Z)[1] + np.linalg.norm(Z)**2/2, p.proximal_optimum(Z)[1]

        d = dual(shape)
        p = d.conjugate
        yield nt.assert_equal, p, primal(shape)
        yield ac, Z-p.proximal(Z), d.proximal(Z)

    #        yield ac, d.proximal_optimum(Z)[1], p.proximal_optimum(Z)[1] + np.linalg.norm(Z)**2/2

def test_linear_term_proximal():
    shape = 20

    Z = np.random.standard_normal(shape)
    W = 0.02 * np.random.standard_normal(shape)
    for primal, dual in C.conjugate_cone_pairs.items():
        p = primal(shape, linear_term=W)
        d = p.conjugate
        print p, d
        yield nt.assert_equal, d, dual(shape)
        yield ac, p.proximal(Z), Z-d.proximal(Z)
        ##yield ac, d.proximal_optimum(Z)[1] + np.linalg.norm(Z)**2/2, p.proximal_optimum(Z)[1]

        d = dual(shape, linear_term=W)
        p = d.conjugate
        yield nt.assert_equal, p, primal(shape)
        yield ac, p.proximal(Z), Z-d.proximal(Z)

        ##        yield ac, d.proximal_optimum(Z)[1], p.proximal_optimum(Z)[1] + np.linalg.norm(Z)**2/2

        loss = rr.quadratic.shift(-Z, coef=0.5)
        problem = rr.composite(loss.smooth_objective, p.nonsmooth_objective,
                               p.proximal, np.random.standard_normal(shape))
        solver = rr.FISTA(problem)
        solver.fit(tol=1.0e-12)

        yield ac, p.proximal(Z), solver.composite.coefs
        problem = rr.composite(loss.smooth_objective, d.nonsmooth_objective,
                               d.proximal, np.random.standard_normal(shape))
        solver = rr.FISTA(problem)
        solver.fit(tol=1.0e-12)
        yield ac, d.proximal(Z), solver.composite.coefs

def test_offset_proximal():
    shape = 50

    Z = np.random.standard_normal(shape)
    W = 0.02 * np.random.standard_normal(shape)
    for primal, dual in C.conjugate_cone_pairs.items():
        p = primal(shape, offset=W)
        d = p.conjugate
        print p, d
        yield nt.assert_equal, d, dual(shape)
        yield ac, Z-p.proximal(Z), d.proximal(Z)
        #yield ac, d.proximal_optimum(Z)[1] + np.linalg.norm(Z)**2/2, p.proximal_optimum(Z)[1]

        d = dual(shape, offset=W)
        p = d.conjugate
        yield nt.assert_equal, p, primal(shape)
        yield ac, Z-p.proximal(Z), d.proximal(Z)
#        yield ac, d.proximal_optimum(Z)[1], p.proximal_optimum(Z)[1] + np.linalg.norm(Z)**2/2

def test_offset_and_linear_term_proximal():
    shape = 50

    Z = np.random.standard_normal(shape)
    W = 0.02 * np.random.standard_normal(shape)
    U = 0.02 * np.random.standard_normal(shape)
    for primal, dual in C.conjugate_cone_pairs.items():
        p = primal(shape, offset=W, linear_term=U)
        d = p.conjugate
        print p, d
        yield nt.assert_equal, d, dual(shape)
        yield ac, Z-p.proximal(Z), d.proximal(Z)
        #yield ac, d.proximal_optimum(Z)[1] + np.linalg.norm(Z)**2/2, p.proximal_optimum(Z)[1]

        d = dual(shape, offset=W, linear_term=U)
        p = d.conjugate
        yield nt.assert_equal, p, primal(shape)
        yield ac, Z-p.proximal(Z), d.proximal(Z)
#        yield ac, d.proximal_optimum(Z)[1], p.proximal_optimum(Z)[1] + np.linalg.norm(Z)**2/2

