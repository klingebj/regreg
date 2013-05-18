import numpy as np
import itertools
from copy import copy

np.random.seed(0)

import regreg.atoms.seminorms as S
import regreg.api as rr
import nose.tools as nt

def all_close(x, y, msg, solver):
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
FAIL
====
msg: %s
comparison: %0.3f
x : %s
y : %s
""" % (msg, np.linalg.norm(x-y) / max([1, np.linalg.norm(x), np.linalg.norm(y)]), x, y))
    if not hasattr(solver, 'interactive') or not solver.interactive:
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
                      S.positive_part, S.constrained_max][:1],
                                              [None, quadratic],
                                              [U, None],
                                              [False, True],
                                              [True, False]):

        p = atom(shape, lagrange=lagrange, quadratic=q,
                   offset=offset)
        d = p.conjugate 
        yield all_close, p.lagrange_prox(Z, lipschitz=L), Z-d.bound_prox(Z*L)/L, 'testing lagrange_prox and bound_prox starting from atom %s ' % atom, None
        # some arguments of the constructor

        nt.assert_raises(AttributeError, setattr, p, 'bound', 4.)
        nt.assert_raises(AttributeError, setattr, d, 'lagrange', 4.)
        
        nt.assert_raises(AttributeError, setattr, p, 'bound', 4.)
        nt.assert_raises(AttributeError, setattr, d, 'lagrange', 4.)

        for t in Solver(p, Z, L, FISTA, coef_stop).all():
            yield t

        b = atom(shape, bound=bound, quadratic=q,
                 offset=offset)

        for t in Solver(b, Z, L, FISTA, coef_stop).all():
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
        yield all_close, p.lagrange_prox(Z, lipschitz=L), Z-d.bound_prox(Z*L)/L, 'testing lagrange_prox and bound_prox starting from atom %s ' % atom, None
        # some arguments of the constructor

        nt.assert_raises(AttributeError, setattr, p, 'bound', 4.)
        nt.assert_raises(AttributeError, setattr, d, 'lagrange', 4.)
        
        nt.assert_raises(AttributeError, setattr, p, 'bound', 4.)
        nt.assert_raises(AttributeError, setattr, d, 'lagrange', 4.)

        for t in Solver(p, Z, L, FISTA, coef_stop).all():
            yield t


class Solver(object):

    def __repr__(self):
        return 'Solver(%s)' % repr(self.atom)

    def __init__(self, atom, Z, L, FISTA=True, coef_stop=True, container=True, interactive=False):
        self.interactive = interactive
        self.coef_stop = coef_stop
        self.atom = atom
        self.Z = Z
        self.L = L
        self.FISTA = FISTA
        self.do_container = container
        self.q = rr.identity_quadratic(L, Z, 0, 0)
        self.loss = rr.quadratic.shift(Z, coef=L)

    def duality_of_projections(self):
        tests = []

        d = self.atom.conjugate
        q = rr.identity_quadratic(1, self.Z, 0, 0)
        tests.append((self.Z-self.atom.proximal(q), d.proximal(q), 'testing duality of projections starting from atom %s ' % str(self)))

        if not self.interactive:
            for test in tests:
                yield (all_close,) + test + (self,)
        else:
            for test in tests:
                yield all_close(*((test + (self,))))

    def simple_problem_nonsmooth(self):
        tests = []
        atom, q = self.atom, self.q
        loss = self.loss

        p2 = copy(atom)
        p2.quadratic = atom.quadratic + q
        problem = rr.simple_problem.nonsmooth(p2)
        solver = rr.FISTA(problem)
        solver.fit(tol=1.0e-14, FISTA=self.FISTA, coef_stop=self.coef_stop)

        tests.append((atom.proximal(q), solver.composite.coefs, 'solving prox with simple_problem.nonsmooth with monotonicity %s ' % str(self)))

        # use the solve method

        p3 = copy(atom)
        p3.quadratic = atom.quadratic + q
        soln = p3.solve(tol=1.e-14, min_its=10)
        tests.append((atom.proximal(q), soln, 'solving prox with solve method %s ' % str(self)))
        print 'stop'

        p4 = copy(atom)
        p4.quadratic = atom.quadratic + q
        problem = rr.simple_problem.nonsmooth(p4)
        solver = rr.FISTA(problem)
        solver.fit(tol=1.0e-14, monotonicity_restart=False, coef_stop=self.coef_stop,
                   FISTA=self.FISTA)

        tests.append((atom.proximal(q), solver.composite.coefs, 'solving prox with simple_problem.nonsmooth with no monotonocity %s ' % str(self)))

        if not self.interactive:
            for test in tests:
                yield (all_close,) + test + (self,)
        else:
            for test in tests:
                yield all_close(*((test + (self,))))

    def simple_problem(self):
        tests = []
        atom, q, Z, L = self.atom, self.q, self.Z, self.L
        loss = self.loss

        problem = rr.simple_problem(loss, atom)
        solver = rr.FISTA(problem)
        solver.fit(tol=1.0e-12, FISTA=self.FISTA, coef_stop=self.coef_stop)

        tests.append((atom.proximal(q), solver.composite.coefs, 'solving prox with simple_problem with monotonicity: %s' % str(self)))

        # write the loss in terms of a quadratic for the smooth loss and a smooth function...

        q = rr.identity_quadratic(L, Z, 0, 0)
        lossq = rr.quadratic.shift(Z.copy(), coef=0.6*L)
        lossq.quadratic = rr.identity_quadratic(0.4*L, Z.copy(), 0, 0)
        problem = rr.simple_problem(lossq, atom)

        tests.append((atom.proximal(q), 
              problem.solve(coef_stop=self.coef_stop, 
                            FISTA=self.FISTA, 
                            tol=1.0e-12), 
               'solving prox with simple_problem ' +
               'with monotonicity  but loss has identity_quadratic %s ' % str(self)))

        problem = rr.simple_problem(loss, atom)
        solver = rr.FISTA(problem)
        solver.fit(tol=1.0e-12, monotonicity_restart=False,
                   coef_stop=self.coef_stop, FISTA=self.FISTA)

        tests.append((atom.proximal(q), solver.composite.coefs, 'solving prox with simple_problem %s no monotonicity_restart' % str(self)))

        d = atom.conjugate
        problem = rr.simple_problem(loss, d)
        solver = rr.FISTA(problem)
        solver.fit(tol=1.0e-12, monotonicity_restart=False, 
                   coef_stop=self.coef_stop, FISTA=self.FISTA)
        tests.append((d.proximal(q), problem.solve(tol=1.e-12,
                                                FISTA=self.FISTA,
                                                coef_stop=self.coef_stop,
                                                monotonicity_restart=False), 
               'solving dual prox with simple_problem no monotonocity %s ' % str(self)))

        if not self.interactive:
            for test in tests:
                yield (all_close,) + test + (self,)
        else:
            for test in tests:
                yield all_close(*((test + (self,))))

    def dual_problem(self):
        tests = []
        atom, q, Z, L = self.atom, self.q, self.Z, self.L
        loss = self.loss

        dproblem = rr.dual_problem.fromprimal(loss, atom)
        dcoef = dproblem.solve(coef_stop=self.coef_stop, tol=1.0e-14)
        tests.append((atom.proximal(q), dcoef, 'solving prox with dual_problem.fromprimal with monotonicity %s ' % str(self)))

        dproblem2 = rr.dual_problem(loss.conjugate, 
                                    rr.identity(loss.shape),
                                    atom.conjugate)
        dcoef2 = dproblem2.solve(coef_stop=self.coef_stop, tol=1.e-14)
        tests.append((atom.proximal(q), dcoef2, 'solving prox with dual_problem with monotonicity %s ' % str(self)))

        if not self.interactive:
            print 'here'
            for test in tests:
                yield (all_close,) + test + (self,)
        else:
            for test in tests:
                yield all_close(*((test + (self,))))

    def separable(self):
        tests = []
        atom, q, Z, L = self.atom, self.q, self.Z, self.L
        loss = self.loss

        problem = rr.separable_problem.singleton(atom, loss)
        solver = rr.FISTA(problem)
        solver.fit(tol=1.0e-12, 
                   coef_stop=self.coef_stop, FISTA=self.FISTA)

        tests.append((atom.proximal(q), solver.composite.coefs, 'solving atom prox with separable_atom.singleton %s ' % str(self)))


        d = atom.conjugate
        problem = rr.separable_problem.singleton(d, loss)
        solver = rr.FISTA(problem)
        solver.fit(tol=1.0e-12, 
                   coef_stop=self.coef_stop, FISTA=self.FISTA)

        tests.append((d.proximal(q), solver.composite.coefs, 'solving atom prox with separable_atom.singleton %s ' % str(self)))

        if not self.interactive:
            for test in tests:
                yield (all_close,) + test + (self,)
        else:
            for test in tests:
                yield all_close(*((test + (self,))))

    def container(self):
        tests = []
        atom, q, Z, L = self.atom, self.q, self.Z, self.L
        loss = self.loss

        problem = rr.container(loss, atom)
        solver = rr.FISTA(problem)
        solver.fit(tol=1.0e-12, 
                   coef_stop=self.coef_stop, FISTA=self.FISTA)

        tests.append((atom.proximal(q), solver.composite.coefs, 'solving atom prox with container %s ' % str(self)))

        # write the loss in terms of a quadratic for the smooth loss and a smooth function...

        lossq = rr.quadratic.shift(Z, coef=0.6*L)
        lossq.quadratic = rr.identity_quadratic(0.4*L, Z, 0, 0)
        problem = rr.container(lossq, atom)
        solver = rr.FISTA(problem)
        solver.fit(tol=1.0e-12, FISTA=self.FISTA, coef_stop=self.coef_stop)

        tests.append((atom.proximal(q), 
               problem.solve(tol=1.e-12,FISTA=self.FISTA,coef_stop=self.coef_stop), 
               'solving prox with container with monotonicity but loss has identity_quadratic %s ' % str(self)))

        d = atom.conjugate
        problem = rr.container(d, loss)
        solver = rr.FISTA(problem)
        solver.fit(tol=1.0e-12, 
                   coef_stop=self.coef_stop, FISTA=self.FISTA)
        tests.append((d.proximal(q), solver.composite.coefs, 'solving dual prox with container %s ' % str(self)))

        if not self.interactive:
            for test in tests:
                yield (all_close,) + test + (self,)
        else:
            for test in tests:
                yield all_close(*((test + (self,))))

    def all(self):
        for group in [self.simple_problem,
                  self.separable,
                  self.dual_problem,
                  self.container,
                  self.simple_problem_nonsmooth,
                  self.duality_of_projections]:
            for t in group():
                yield t
                  

