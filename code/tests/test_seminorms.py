import numpy as np
import itertools
from copy import copy

import regreg.atoms as A
import regreg.cones as C
import regreg.linear_constraints as LC
import regreg.api as rr
import nose.tools as nt

def ac(x,y, msg=None):
    v = np.linalg.norm(x-y) <= 1.0e-04 * max([1, np.linalg.norm(x), np.linalg.norm(y)])
    if not (v or np.allclose(x,y)):
        print 'msg: ', msg, np.linalg.norm(x-y) / max([1, np.linalg.norm(x), np.linalg.norm(y)]), x, y
    nt.assert_true(v)

@np.testing.dec.slow
def test_proximal_maps():
    bound = 0.14
    lagrange = 0.13
    shape = 20

    Z = np.random.standard_normal(shape) * 2
    W = 0.02 * np.random.standard_normal(shape)
    U = 0.02 * np.random.standard_normal(shape)
    linq = rr.identity_quadratic(0,0,W,0)

    for L, atom, q, offset, FISTA in itertools.product([0.5,1,0.1], \
                     sorted(A.conjugate_seminorm_pairs.keys()),
                                              [None, linq],
                                              [None, U],
                                              [False, True]):

        p = atom(shape, lagrange=lagrange, quadratic=q,
                   offset=offset)
        d = p.conjugate 
        yield ac, p.lagrange_prox(Z, lipschitz=L), Z-d.bound_prox(Z*L, lipschitz=1./L)/L, 'testing lagrange_prox and bound_prox starting from atom %s ' % atom
        # some arguments of the constructor

        nt.assert_raises(AttributeError, setattr, p, 'bound', 4.)
        nt.assert_raises(AttributeError, setattr, d, 'lagrange', 4.)
        
        nt.assert_raises(AttributeError, setattr, p, 'bound', 4.)
        nt.assert_raises(AttributeError, setattr, d, 'lagrange', 4.)

        for t in solveit(p, Z, W, U, linq, L, FISTA):
            yield t

        b = atom(shape, bound=bound, quadratic=q,
                 offset=offset)

        for t in solveit(b, Z, W, U, linq, L, FISTA):
            yield t


    for L, atom, q, offset, FISTA in itertools.product([0.5,1,0.1], \
                     sorted(A.nonpaired_atoms),
                                              [None, linq],
                                              [None, U],
                                              [False, True]):

        p = atom(shape, lagrange=lagrange, quadratic=q,
                   offset=offset)
        d = p.conjugate 
        yield ac, p.lagrange_prox(Z, lipschitz=L), Z-d.bound_prox(Z*L, lipschitz=1./L)/L, 'testing lagrange_prox and bound_prox starting from atom %s ' % atom
        # some arguments of the constructor

        nt.assert_raises(AttributeError, setattr, p, 'bound', 4.)
        nt.assert_raises(AttributeError, setattr, d, 'lagrange', 4.)
        
        nt.assert_raises(AttributeError, setattr, p, 'bound', 4.)
        nt.assert_raises(AttributeError, setattr, d, 'lagrange', 4.)

        for t in solveit(p, Z, W, U, linq, L, FISTA):
            yield t


def solveit(atom, Z, W, U, linq, L, FISTA):

        p2 = copy(atom)
        p2.set_quadratic(L, Z, 0, 0)

        d = atom.conjugate

        q = rr.identity_quadratic(1, Z, 0, 0)
        yield ac, Z-atom.proximal(q), d.proximal(q), 'testing duality of projections starting from atom %s ' % atom
        q = rr.identity_quadratic(L, Z, 0, 0)

        p2 = copy(atom)
        p2.set_quadratic(L, Z, 0, 0)
        problem = rr.simple_problem.nonsmooth(p2)
        solver = rr.FISTA(problem)
        solver.fit(tol=1.0e-14, FISTA=FISTA)

        # yield ac, atom.proximal(q), solver.composite.coefs, 'solving prox with simple_problem.nonsmooth with monotonicity %s ' % atom

        loss = rr.quadratic.shift(-Z, coef=0.5*L)
        problem = rr.simple_problem(loss, atom)
        solver = rr.FISTA(problem)

        # restarting is acting funny
        solver.fit(tol=1.0e-12, FISTA=FISTA)

        yield ac, atom.proximal(q), solver.composite.coefs, 'solving prox with simple_problem with monotonicity %s ' % atom

        problem = rr.simple_problem.nonsmooth(p2)
        solver = rr.FISTA(problem)
        solver.fit(tol=1.0e-14, monotonicity_restart=False, FISTA=FISTA)

        # yield ac, atom.proximal(q), solver.composite.coefs, 'solving prox with simple_problem.nonsmooth with no monotonocity %s ' % atom

        loss = rr.quadratic.shift(-Z, coef=0.5*L)
        problem = rr.simple_problem(loss, atom)
        solver = rr.FISTA(problem)
        solver.fit(tol=1.0e-12, monotonicity_restart=False, FISTA=FISTA)

        yield ac, atom.proximal(q), solver.composite.coefs, 'solving prox with simple_problem %s no monotonicity_restart' % atom

        loss = rr.quadratic.shift(-Z, coef=0.5*L)
        problem = rr.separable_problem.singleton(atom, loss)
        solver = rr.FISTA(problem)
        solver.fit(tol=1.0e-12, FISTA=FISTA)

        yield ac, atom.proximal(q), solver.composite.coefs, 'solving atom prox with separable_atom.singleton %s ' % atom

        loss = rr.quadratic.shift(-Z, coef=0.5*L)
        problem = rr.container(loss, atom)
        solver = rr.FISTA(problem)
        solver.fit(tol=1.0e-12, FISTA=FISTA)

        yield ac, atom.proximal(q), solver.composite.coefs, 'solving atom prox with container %s ' % atom

        loss = rr.quadratic.shift(-Z, coef=0.5*L)
        problem = rr.simple_problem(loss, d)
        solver = rr.FISTA(problem)
        solver.fit(tol=1.0e-12, monotonicity_restart=False, FISTA=FISTA)
        yield ac, d.proximal(q), solver.composite.coefs, 'solving dual prox with simple_problem no monotonocity %s ' % atom

        problem = rr.container(d, loss)
        solver = rr.FISTA(problem)
        solver.fit(tol=1.0e-12, FISTA=FISTA)
        yield ac, d.proximal(q), solver.composite.coefs, 'solving dual prox with container %s ' % atom

        loss = rr.quadratic.shift(-Z, coef=0.5*L)
        problem = rr.separable_problem.singleton(d, loss)
        solver = rr.FISTA(problem)
        solver.fit(tol=1.0e-12, FISTA=FISTA)

        yield ac, d.proximal(q), solver.composite.coefs, 'solving atom prox with separable_atom.singleton %s ' % atom


