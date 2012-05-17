import numpy as np
import itertools

import regreg.atoms as A
import regreg.api as rr
import nose.tools as nt

def ac(x,y, msg=None):
    v = np.linalg.norm(x-y) <= 1.0e-04 * max([1, np.linalg.norm(x), np.linalg.norm(y)])
    if not v:
        print 'msg: ', msg, np.linalg.norm(x-y) / max([1, np.linalg.norm(x), np.linalg.norm(y)])
    nt.assert_true(v)

def test_proximal_maps():
    bound = 0.14
    lagrange = 0.13
    shape = 20

    Z = np.random.standard_normal(shape) * 2
    W = 0.02 * np.random.standard_normal(shape)
    U = 0.02 * np.random.standard_normal(shape)
    linq = rr.identity_quadratic(0,0,W,0)

    for L, pd, q, offset, FISTA in itertools.product([0.5,1,0.1], \
                     sorted(A.conjugate_seminorm_pairs.items()),
                                              [None, linq],
                                              [None, U],
                                              [False, True]):
        primal, dual = pd

        p = primal(shape, lagrange=lagrange, quadratic=q,
                   offset=offset)
        d = p.conjugate

        yield ac, Z-p.proximal(1, Z, 0), d.proximal(1, Z, 0), 'testing dual of projections starting from primal %s, %s' % (primal, dual)
        yield ac, p.lagrange_prox(Z, lipschitz=L), Z-d.bound_prox(Z*L, lipschitz=1./L)/L, 'testing lagrange_prox and bound_prox starting from primal %s, %s' % (primal, dual)

        # some arguments of the constructor

        nt.assert_raises(AttributeError, setattr, p, 'bound', 4.)
        nt.assert_raises(AttributeError, setattr, d, 'lagrange', 4.)

        nt.assert_raises(AttributeError, setattr, p, 'bound', 4.)
        nt.assert_raises(AttributeError, setattr, d, 'lagrange', 4.)

        p2 = primal(shape, lagrange=lagrange)
        p2.set_quadratic(L, Z, 0, 0)
        problem = rr.simple_problem.nonsmooth(p2)
        solver = rr.FISTA(problem)
        solver.fit(tol=1.0e-14, FISTA=FISTA)

        yield ac, p.proximal(L, Z, 0), solver.composite.coefs, 'solving prox with simple_problem.nonsmooth with monotonicity %s, %s' % (primal, dual)

        loss = rr.quadratic.shift(-Z, coef=0.5*L)
        problem = rr.simple_problem(loss, p)
        solver = rr.FISTA(problem)

        # restarting is acting funny
        solver.fit(tol=1.0e-12, FISTA=FISTA)

        yield ac, p.proximal(L, Z, 0), solver.composite.coefs, 'solving prox with simple_problem with monotonicity %s, %s' % (primal, dual)

        p2 = primal(shape, lagrange=lagrange)
        p2.set_quadratic(L, Z, 0, 0)
        problem = rr.simple_problem.nonsmooth(p2)
        solver = rr.FISTA(problem)
        solver.fit(tol=1.0e-14, monotonicity_restart=False, FISTA=FISTA)

        yield ac, p.proximal(L, Z, 0), solver.composite.coefs, 'solving prox with simple_problem.nonsmooth with no monotonocity %s, %s' % (primal, dual)

        loss = rr.quadratic.shift(-Z, coef=0.5*L)
        problem = rr.simple_problem(loss, p)
        solver = rr.FISTA(problem)
        solver.fit(tol=1.0e-12, monotonicity_restart=False, FISTA=FISTA)

        yield ac, p.proximal(L, Z, 0), solver.composite.coefs, 'solving prox with simple_problem %s, %s no monotonicity_restart' % (primal, dual)

        loss = rr.quadratic.shift(-Z, coef=0.5*L)
        problem = rr.separable_problem.singleton(p, loss)
        solver = rr.FISTA(problem)
        solver.fit(tol=1.0e-12, FISTA=FISTA)

        yield ac, p.proximal(L, Z, 0), solver.composite.coefs, 'solving primal prox with separable_atom.singleton %s, %s' % (primal, dual)

        loss = rr.quadratic.shift(-Z, coef=0.5*L)
        problem = rr.container(loss, p)
        solver = rr.FISTA(problem)
        solver.fit(tol=1.0e-12, FISTA=FISTA)

        yield ac, p.proximal(L, Z, 0), solver.composite.coefs, 'solving primal prox with container %s, %s' % (primal, dual)

        loss = rr.quadratic.shift(-Z, coef=0.5*L)
        problem = rr.simple_problem(loss, d)
        solver = rr.FISTA(problem)
        solver.fit(tol=1.0e-12, monotonicity_restart=False, FISTA=FISTA)
        yield ac, d.proximal(L, Z, 0), solver.composite.coefs, 'solving dual prox with simple_problem no monotonocity %s, %s' % (primal, dual)

        problem = rr.container(d, loss)
        solver = rr.FISTA(problem)
        solver.fit(tol=1.0e-12, FISTA=FISTA)
        yield ac, d.proximal(L, Z, 0), solver.composite.coefs, 'solving dual prox with container %s, %s' % (primal, dual)

        loss = rr.quadratic.shift(-Z, coef=0.5*L)
        problem = rr.separable_problem.singleton(d, loss)
        solver = rr.FISTA(problem)
        solver.fit(tol=1.0e-12, FISTA=FISTA)

        yield ac, d.proximal(L, Z, 0), solver.composite.coefs, 'solving primal prox with separable_atom.singleton %s, %s' % (primal, dual)


