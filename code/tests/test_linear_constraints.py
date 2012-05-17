import numpy as np
import regreg.linear_constraints as C
import regreg.api as rr
import nose.tools as nt

def ac(x,y):
    print x, y
    return nt.assert_true(np.linalg.norm(x-y) <= 1.0e-04 * max([1, np.linalg.norm(x), np.linalg.norm(y)]))

def test_proximal_maps():
    shape = 20

    basis = np.linalg.svd(np.random.standard_normal((shape,shape)))[2][:5]
    Z = np.random.standard_normal(shape)
    for primal, dual in C.conjugate_cone_pairs.items():
        p = primal(shape, basis)
        d = p.conjugate
        yield nt.assert_equal, d, dual(shape, basis)
        yield ac, Z-p.proximal(1, Z, 0), d.proximal(1, Z, 0)

        loss = rr.quadratic.shift(-Z, coef=0.5)
        problem = rr.simple_problem(loss, p)
        solver = rr.FISTA(problem)
        solver.fit(tol=1.0e-12)

        yield ac, p.proximal(1, Z, 0), solver.composite.coefs

        loss = rr.quadratic.shift(-Z, coef=0.5)
        problem = rr.container(loss, p)
        solver = rr.FISTA(problem)
        solver.fit(tol=1.0e-12)

        yield ac, p.proximal(1, Z, 0), solver.composite.coefs

        problem = rr.simple_problem(loss, d)
        solver = rr.FISTA(problem)
        solver.fit(tol=1.0e-12)
        yield ac, d.proximal(1, Z, 0), solver.composite.coefs

        #yield ac, d.proximal_optimum(Z)[1] + np.linalg.norm(Z)**2/2, p.proximal_optimum(Z)[1]

        d = dual(shape, basis)
        p = d.conjugate
        yield nt.assert_equal, p, primal(shape, basis)
        yield ac, Z-p.proximal(1, Z, 0), d.proximal(1, Z, 0)

    #        yield ac, d.proximal_optimum(Z)[1], p.proximal_optimum(Z)[1] + np.linalg.norm(Z)**2/2

def test_linear_term_proximal():
    shape = 20

    Z = np.random.standard_normal(shape)
    W = 0.02 * np.random.standard_normal(shape)
    basis = np.linalg.svd(np.random.standard_normal((shape,shape)))[2][:5]
    linq = rr.identity_quadratic(0,0,W,0)

    for primal, dual in C.conjugate_cone_pairs.items():

        p = primal(shape, basis, quadratic=linq)
        d = p.conjugate
        print p, d
        yield nt.assert_equal, d, dual(shape, basis)
        yield ac, p.proximal(1, Z, 0), Z-d.proximal(1, Z, 0)
        ##yield ac, d.proximal_optimum(Z)[1] + np.linalg.norm(Z)**2/2, p.proximal_optimum(Z)[1]

        d = dual(shape, basis, quadratic=linq)
        p = d.conjugate
        yield nt.assert_equal, p, primal(shape, basis)
        yield ac, p.proximal(1, Z, 0), Z-d.proximal(1, Z, 0)

        ##        yield ac, d.proximal_optimum(Z)[1], p.proximal_optimum(Z)[1] + np.linalg.norm(Z)**2/2

        loss = rr.quadratic.shift(-Z, coef=0.5)
        problem = rr.simple_problem(loss, p)
        solver = rr.FISTA(problem)
        solver.fit(tol=1.0e-12)

        yield ac, p.proximal(1, Z, 0), solver.composite.coefs
        problem = rr.simple_problem(loss, d)
        solver = rr.FISTA(problem)
        solver.fit(tol=1.0e-12)
        yield ac, d.proximal(1, Z, 0), solver.composite.coefs

def test_offset_proximal():
    shape = 50

    basis = np.linalg.svd(np.random.standard_normal((shape,shape)))[2][:5]

    Z = np.random.standard_normal(shape)
    W = 0.02 * np.random.standard_normal(shape)
    for primal, dual in C.conjugate_cone_pairs.items():
        p = primal(shape, basis, offset=W)
        d = p.conjugate
        print p, d
        yield nt.assert_equal, d, dual(shape, basis)
        yield ac, Z-p.proximal(1, Z, 0), d.proximal(1, Z, 0)
        #yield ac, d.proximal_optimum(Z)[1] + np.linalg.norm(Z)**2/2, p.proximal_optimum(Z)[1]

        d = dual(shape, basis, offset=W)
        p = d.conjugate
        yield nt.assert_equal, p, primal(shape, basis)
        yield ac, Z-p.proximal(1, Z, 0), d.proximal(1, Z, 0)
#        yield ac, d.proximal_optimum(Z)[1], p.proximal_optimum(Z)[1] + np.linalg.norm(Z)**2/2

def test_offset_and_linear_term_proximal():
    shape = 50

    basis = np.linalg.svd(np.random.standard_normal((shape,shape)))[2][:5]

    Z = np.random.standard_normal(shape)
    W = 0.02 * np.random.standard_normal(shape)
    U = 0.02 * np.random.standard_normal(shape)
    for primal, dual in C.conjugate_cone_pairs.items():
        p = primal(shape, basis, offset=W, quadratic=linq)
        d = p.conjugate
        print p, d
        yield nt.assert_equal, d, dual(shape, basis)
        yield ac, Z-p.proximal(1, Z, 0), d.proximal(1, Z, 0)
        #yield ac, d.proximal_optimum(Z)[1] + np.linalg.norm(Z)**2/2, p.proximal_optimum(Z)[1]

        d = dual(shape, basis, offset=W, quadratic=linq)
        p = d.conjugate
        yield nt.assert_equal, p, primal(shape, basis)
        yield ac, Z-p.proximal(1, Z, 0), d.proximal(1, Z, 0)
#        yield ac, d.proximal_optimum(Z)[1], p.proximal_optimum(Z)[1] + np.linalg.norm(Z)**2/2

