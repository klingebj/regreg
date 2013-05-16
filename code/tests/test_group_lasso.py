import numpy as np
import itertools
from copy import copy

from regreg.atoms.group_lasso import group_lasso, group_lasso_dual
import regreg.api as rr
import nose.tools as nt

def ac(x,y, msg=None):
    try:
        v = np.linalg.norm(x-y) <= 1.0e-04 * max([1, np.linalg.norm(x), np.linalg.norm(y)])
    except:
        print 'check failed for msg: ', msg, x, y
        stop
        return False
    if not (v or np.allclose(x,y)):
        print 'msg: ', msg, np.linalg.norm(x-y) / max([1, np.linalg.norm(x), np.linalg.norm(y)]), x, y
    nt.assert_true(v)

@np.testing.dec.slow
def test_proximal_maps():
    bound = 0.14
    lagrange = 0.13
    shape = 20

    groups = [0]*3 + [1]*4 + [2]*5 + [3]*8

    Z = np.random.standard_normal(shape) * 4
    W = 0.02 * np.random.standard_normal(shape)
    U = 0.02 * np.random.standard_normal(shape)
    linq = rr.identity_quadratic(0,0,W,0)

    for L, atom, q, offset, FISTA, coef_stop, weights in itertools.product([0.5,1,0.1], \
                     [group_lasso, group_lasso_dual],
                                              [None, linq],
                                              [None, U],
                                              [False, True],
                                              [True, False],
                                              [{}, {0:3}]):

        p = atom(groups, lagrange=lagrange, quadratic=q,
                   offset=offset, weights=weights)
        d = p.conjugate 
        yield ac, p.lagrange_prox(Z, lipschitz=L), Z-d.bound_prox(Z*L, lipschitz=1./L)/L, 'testing lagrange_prox and bound_prox starting from atom %s ' % atom
        # some arguments of the constructor

        nt.assert_raises(AttributeError, setattr, p, 'bound', 4.)
        nt.assert_raises(AttributeError, setattr, d, 'lagrange', 4.)
        
        nt.assert_raises(AttributeError, setattr, p, 'bound', 4.)
        nt.assert_raises(AttributeError, setattr, d, 'lagrange', 4.)

        for t in solveit(p, Z, W, U, linq, L, FISTA, coef_stop):
            yield t

        b = atom(groups, bound=bound, quadratic=q,
                 offset=offset, weights=weights)

        for t in solveit(b, Z, W, U, linq, L, FISTA, coef_stop):
            yield t


    lagrange = 0.1

def solveit(atom, Z, W, U, linq, L, FISTA, coef_stop):

    p2 = copy(atom)
    p2.quadratic = rr.identity_quadratic(L, Z, 0, 0)

    d = atom.conjugate

    q = rr.identity_quadratic(1, Z, 0, 0)
    yield ac, Z-atom.proximal(q), d.proximal(q), 'testing duality of projections starting from atom %s ' % atom
    q = rr.identity_quadratic(L, Z, 0, 0)

    # use simple_problem.nonsmooth

    p2 = copy(atom)
    p2.quadratic = atom.quadratic + q
    problem = rr.simple_problem.nonsmooth(p2)
    solver = rr.FISTA(problem)
    solver.fit(tol=1.0e-14, FISTA=FISTA, coef_stop=coef_stop)

    yield ac, atom.proximal(q), solver.composite.coefs, 'solving prox with simple_problem.nonsmooth with monotonicity %s ' % atom

    # use the solve method

    p2.coefs *= 0
    p2.quadratic = atom.quadratic + q
    soln = p2.solve()

    yield ac, atom.proximal(q), soln, 'solving prox with solve method %s ' % atom

    loss = rr.quadratic.shift(-Z, coef=L)
    problem = rr.simple_problem(loss, atom)
    solver = rr.FISTA(problem)
    solver.fit(tol=1.0e-12, FISTA=FISTA, coef_stop=coef_stop)

    yield ac, atom.proximal(q), solver.composite.coefs, 'solving prox with simple_problem with monotonicity %s ' % atom

    dproblem2 = rr.dual_problem(loss.conjugate, 
                                rr.identity(loss.shape),
                                atom.conjugate)
    dcoef2 = dproblem2.solve(coef_stop=coef_stop, tol=1.e-14)
    yield ac, atom.proximal(q), dcoef2, 'solving prox with dual_problem with monotonicity %s ' % atom

    dproblem = rr.dual_problem.fromprimal(loss, atom)
    dcoef = dproblem.solve(coef_stop=coef_stop, tol=1.0e-14)
    yield ac, atom.proximal(q), dcoef, 'solving prox with dual_problem.fromprimal with monotonicity %s ' % atom

    # write the loss in terms of a quadratic for the smooth loss and a smooth function...

    lossq = rr.quadratic.shift(-Z, coef=0.6*L)
    lossq.quadratic = rr.identity_quadratic(0.4 * L, Z, 0, 0)
    problem = rr.simple_problem(lossq, atom)

    yield ac, atom.proximal(q), problem.solve(coef_stop=coef_stop, FISTA=FISTA, 
                                              tol=1.0e-12), 'solving prox with simple_problem with monotonicity  but loss has identity_quadratic %s ' % atom

    problem = rr.simple_problem.nonsmooth(p2)
    solver = rr.FISTA(problem)
    solver.fit(tol=1.0e-14, monotonicity_restart=False, coef_stop=coef_stop,
               FISTA=FISTA)

    yield ac, atom.proximal(q), solver.composite.coefs, 'solving prox with simple_problem.nonsmooth with no monotonocity %s ' % atom

    loss = rr.quadratic.shift(-Z, coef=L)
    problem = rr.simple_problem(loss, atom)
    solver = rr.FISTA(problem)
    solver.fit(tol=1.0e-12, monotonicity_restart=False,
               coef_stop=coef_stop, FISTA=FISTA)

    yield ac, atom.proximal(q), solver.composite.coefs, 'solving prox with simple_problem %s no monotonicity_restart' % atom

    loss = rr.quadratic.shift(-Z, coef=L)
    problem = rr.separable_problem.singleton(atom, loss)
    solver = rr.FISTA(problem)
    solver.fit(tol=1.0e-12, 
               coef_stop=coef_stop, FISTA=FISTA)

    yield ac, atom.proximal(q), solver.composite.coefs, 'solving atom prox with separable_atom.singleton %s ' % atom

    loss = rr.quadratic.shift(-Z, coef=L)
    problem = rr.container(loss, atom)
    solver = rr.FISTA(problem)
    solver.fit(tol=1.0e-12, 
               coef_stop=coef_stop, FISTA=FISTA)

    yield ac, atom.proximal(q), solver.composite.coefs, 'solving atom prox with container %s ' % atom

    # write the loss in terms of a quadratic for the smooth loss and a smooth function...

    lossq = rr.quadratic.shift(-Z, coef=0.6*L)
    lossq.quadratic = rr.identity_quadratic(0.4 * L, Z, 0, 0)
    problem = rr.container(lossq, atom)
    solver = rr.FISTA(problem)
    solver.fit(tol=1.0e-12, FISTA=FISTA, coef_stop=coef_stop)

    yield (ac, atom.proximal(q), 
           problem.solve(tol=1.e-12,FISTA=FISTA,coef_stop=coef_stop), 
           'solving prox with container with monotonicity  but loss has identity_quadratic %s ' % atom)

    loss = rr.quadratic.shift(-Z, coef=L)
    problem = rr.simple_problem(loss, d)
    solver = rr.FISTA(problem)
    solver.fit(tol=1.0e-12, monotonicity_restart=False, 
               coef_stop=coef_stop, FISTA=FISTA)
    # ac(d.proximal(q), solver.composite.coefs, 'solving dual prox with simple_problem no monotonocity %s ' % atom)
    yield (ac, d.proximal(q), problem.solve(tol=1.e-12,
                                            FISTA=FISTA,
                                            coef_stop=coef_stop,
                                            monotonicity_restart=False), 
           'solving dual prox with simple_problem no monotonocity %s ' % atom)

    problem = rr.container(d, loss)
    solver = rr.FISTA(problem)
    solver.fit(tol=1.0e-12, 
               coef_stop=coef_stop, FISTA=FISTA)
    yield ac, d.proximal(q), solver.composite.coefs, 'solving dual prox with container %s ' % atom

    loss = rr.quadratic.shift(-Z, coef=L)
    problem = rr.separable_problem.singleton(d, loss)
    solver = rr.FISTA(problem)
    solver.fit(tol=1.0e-12, 
               coef_stop=coef_stop, FISTA=FISTA)

    yield ac, d.proximal(q), solver.composite.coefs, 'solving atom prox with separable_atom.singleton %s ' % atom


