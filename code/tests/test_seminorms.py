import numpy as np
import itertools
from copy import copy

import regreg.atoms.seminorms as S
import regreg.api as rr
import nose.tools as nt

def all_close(x,y, msg=None):
    """
    Check to see if x and y are close
    """
    try:
        v = np.linalg.norm(x-y) <= 1.0e-04 * max([1, np.linalg.norm(x), np.linalg.norm(y)])
    except:
        print("""
check_failed
============
msg: %s
x: %s
y: %s
""" % (msg, x, y))
        return False
    v = v or np.allclose(x,y)
    if not v:
        print("""
msg: %s
comparison: %0.3f
x : %s
y : %s
""" % (msg, np.linalg.norm(x-y) / max([1, np.linalg.norm(x), np.linalg.norm(y)]), x, y))
    nt.assert_true(v)

@np.testing.dec.slow
def test_proximal_maps():
    bound = 0.14
    lagrange = 0.13
    shape = 20

    Z = np.random.standard_normal(shape) * 4
    W = 0.02 * np.random.standard_normal(shape)
    U = 0.02 * np.random.standard_normal(shape)
    quadratic = rr.identity_quadratic(0,0,W,0)

    counter = 0
    for L, atom, q, offset, FISTA, coef_stop in itertools.product([0.5,1,0.1], \
                     [S.l1norm, S.supnorm, S.l2norm,
                      S.positive_part, S.constrained_max],
                                              [None, quadratic],
                                              [None, U],
                                              [False, True],
                                              [True, False]):

        p = atom(shape, lagrange=lagrange, quadratic=q,
                   offset=offset)
        d = p.conjugate 
        yield all_close, p.lagrange_prox(Z, lipschitz=L), Z-d.bound_prox(Z*L)/L, 'testing lagrange_prox and bound_prox starting from atom %s ' % atom
        # some arguments of the constructor

        nt.assert_raises(AttributeError, setattr, p, 'bound', 4.)
        nt.assert_raises(AttributeError, setattr, d, 'lagrange', 4.)
        
        nt.assert_raises(AttributeError, setattr, p, 'bound', 4.)
        nt.assert_raises(AttributeError, setattr, d, 'lagrange', 4.)

        for t in solveit(p, Z, quadratic, L, FISTA, coef_stop):
            yield t

        b = atom(shape, bound=bound, quadratic=q,
                 offset=offset)

        stop
        for t in solveit(b, Z, quadratic, L, FISTA, coef_stop):
            yield t


    lagrange = 0.1
    for L, atom, q, offset, FISTA, coef_stop in itertools.product([0.5,1,0.1], \
                     sorted(S.nonpaired_atoms),
                                              [None, quadratic],
                                              [None, U],
                                              [False, True],
                                              [False, True]):

        p = atom(shape, lagrange=lagrange, quadratic=q,
                   offset=offset)
        d = p.conjugate 
        yield all_close, p.lagrange_prox(Z, lipschitz=L), Z-d.bound_prox(Z*L)/L, 'testing lagrange_prox and bound_prox starting from atom %s ' % atom
        # some arguments of the constructor

        nt.assert_raises(AttributeError, setattr, p, 'bound', 4.)
        nt.assert_raises(AttributeError, setattr, d, 'lagrange', 4.)
        
        nt.assert_raises(AttributeError, setattr, p, 'bound', 4.)
        nt.assert_raises(AttributeError, setattr, d, 'lagrange', 4.)

        for t in solveit(p, Z, quadratic, L, FISTA, coef_stop):
            yield t


def solveit(atom, Z, quadratic, L, FISTA, coef_stop):

    args = {'atom':atom,
            'Z':Z,
            'FISTA':FISTA,
            'coef_stop': coef_stop,
            'L':L}

    p2 = copy(atom)
    p2.quadratic = rr.identity_quadratic(L, Z, 0, 0)

    d = atom.conjugate

    q = rr.identity_quadratic(1, Z, 0, 0)
    yield all_close, Z-atom.proximal(q), d.proximal(q), 'testing duality of projections starting from atom %s ' % str(args)
    q = rr.identity_quadratic(L, Z, 0, 0)

    # use simple_problem.nonsmooth

    p2 = copy(atom)
    p2.quadratic = atom.quadratic + q
    problem = rr.simple_problem.nonsmooth(p2)
    solver = rr.FISTA(problem)
    solver.fit(tol=1.0e-14, FISTA=FISTA, coef_stop=coef_stop)

    yield all_close, atom.proximal(q), solver.composite.coefs, 'solving prox with simple_problem.nonsmooth with monotonicity %s ' % str(args)

    # use the solve method

    p2.coefs *= 0
    p2.quadratic = atom.quadratic + q
    soln = p2.solve()

    yield all_close, atom.proximal(q), soln, 'solving prox with solve method %s ' % str(args)

    loss = rr.quadratic.shift(Z, coef=L)
    problem = rr.simple_problem(loss, atom)
    solver = rr.FISTA(problem)
    solver.fit(tol=1.0e-12, FISTA=FISTA, coef_stop=coef_stop)

    yield all_close, atom.proximal(q), solver.composite.coefs, 'solving prox with simple_problem with monotonicity %s ' % str(args)

    dproblem2 = rr.dual_problem(loss.conjugate, 
                                rr.identity(loss.shape),
                                atom.conjugate)
    dcoef2 = dproblem2.solve(coef_stop=coef_stop, tol=1.e-14)
    yield all_close, atom.proximal(q), dcoef2, 'solving prox with dual_problem with monotonicity %s ' % str(args)

    dproblem = rr.dual_problem.fromprimal(loss, atom)
    dcoef = dproblem.solve(coef_stop=coef_stop, tol=1.0e-14)
    yield all_close, atom.proximal(q), dcoef, 'solving prox with dual_problem.fromprimal with monotonicity %s ' % str(args)

    # write the loss in terms of a quadratic for the smooth loss and a smooth function...

    q = rr.identity_quadratic(L, Z, 0, 0)
    lossq = rr.quadratic.shift(Z.copy(), coef=0.6*L)
    lossq.quadratic = rr.identity_quadratic(0.4*L, Z.copy(), 0, 0)
    problem = rr.simple_problem(lossq, atom)

    yield (all_close, atom.proximal(q), 
          problem.solve(coef_stop=coef_stop, 
                        FISTA=FISTA, 
                        tol=1.0e-12), 
           'solving prox with simple_problem ' +
           'with monotonicity  but loss has identity_quadratic %s ' % str(args))

    problem = rr.simple_problem.nonsmooth(p2)
    solver = rr.FISTA(problem)
    solver.fit(tol=1.0e-14, monotonicity_restart=False, coef_stop=coef_stop,
               FISTA=FISTA)

    yield all_close, atom.proximal(q), solver.composite.coefs, 'solving prox with simple_problem.nonsmooth with no monotonocity %s ' % str(args)

    loss = rr.quadratic.shift(Z, coef=L)
    problem = rr.simple_problem(loss, atom)
    solver = rr.FISTA(problem)
    solver.fit(tol=1.0e-12, monotonicity_restart=False,
               coef_stop=coef_stop, FISTA=FISTA)

    yield all_close, atom.proximal(q), solver.composite.coefs, 'solving prox with simple_problem %s no monotonicity_restart' % str(args)

    loss = rr.quadratic.shift(Z, coef=L)
    problem = rr.separable_problem.singleton(atom, loss)
    solver = rr.FISTA(problem)
    solver.fit(tol=1.0e-12, 
               coef_stop=coef_stop, FISTA=FISTA)

    yield all_close, atom.proximal(q), solver.composite.coefs, 'solving atom prox with separable_atom.singleton %s ' % str(args)

    loss = rr.quadratic.shift(Z, coef=L)
    problem = rr.container(loss, atom)
    solver = rr.FISTA(problem)
    solver.fit(tol=1.0e-12, 
               coef_stop=coef_stop, FISTA=FISTA)

    yield all_close, atom.proximal(q), solver.composite.coefs, 'solving atom prox with container %s ' % str(args)

    # write the loss in terms of a quadratic for the smooth loss and a smooth function...

    lossq = rr.quadratic.shift(Z, coef=0.6*L)
    lossq.quadratic = rr.identity_quadratic(0.4 * L, Z, 0, 0)
    problem = rr.container(lossq, atom)
    solver = rr.FISTA(problem)
    solver.fit(tol=1.0e-12, FISTA=FISTA, coef_stop=coef_stop)

    yield (all_close, atom.proximal(q), 
           problem.solve(tol=1.e-12,FISTA=FISTA,coef_stop=coef_stop), 
           '%s solving prox with container with monotonicity  but loss has identity_quadratic %s ' % (lossq, str(args)))

    loss = rr.quadratic.shift(Z, coef=L)
    problem = rr.simple_problem(loss, d)
    solver = rr.FISTA(problem)
    solver.fit(tol=1.0e-12, monotonicity_restart=False, 
               coef_stop=coef_stop, FISTA=FISTA)
    # all_close(d.proximal(q), solver.composite.coefs, 'solving dual prox with simple_problem no monotonocity %s ' % str(args))
    yield (all_close, d.proximal(q), problem.solve(tol=1.e-12,
                                            FISTA=FISTA,
                                            coef_stop=coef_stop,
                                            monotonicity_restart=False), 
           'solving dual prox with simple_problem no monotonocity %s ' % str(args))

    problem = rr.container(d, loss)
    solver = rr.FISTA(problem)
    solver.fit(tol=1.0e-12, 
               coef_stop=coef_stop, FISTA=FISTA)
    yield all_close, d.proximal(q), solver.composite.coefs, 'solving dual prox with container %s ' % str(args)

    loss = rr.quadratic.shift(Z, coef=L)
    problem = rr.separable_problem.singleton(d, loss)
    solver = rr.FISTA(problem)
    solver.fit(tol=1.0e-12, 
               coef_stop=coef_stop, FISTA=FISTA)

    yield all_close, d.proximal(q), solver.composite.coefs, 'solving atom prox with separable_atom.singleton %s ' % str(args)


