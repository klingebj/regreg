
import numpy as np

from copy import copy
import scipy.optimize

import regreg.api as rr
from test_seminorms import ac

def test_lasso():
    '''
    this test verifies that the l1 prox can be solved
    by a primal/dual specification 

    obviously, we don't to solve the l1 prox this way,
    but it verifies that specification is working correctly

    '''

    l1 = rr.l1norm(4, lagrange=2.)

    l11 = rr.l1norm(4, lagrange=1.)
    l12 = rr.l1norm(4, lagrange=1.)


    X = np.random.standard_normal((10,4))
    Y = np.random.standard_normal(10) + 3
    
    loss = rr.quadratic.affine(X, -Y)

    p1 = rr.container(l11, loss, l12)

    solver1 = rr.FISTA(p1)
    solver1.fit(tol=1.0e-12, min_its=500)

    p2 = rr.separable_problem.singleton(l1, loss)
    solver2 = rr.FISTA(p2)
    solver2.fit(tol=1.0e-12)

    f = p2.objective
    ans = scipy.optimize.fmin_powell(f, np.zeros(4), ftol=1.0e-12)
    print f(solver2.composite.coefs), f(ans)
    print f(solver1.composite.coefs), f(ans)

    yield ac, ans, solver2.composite.coefs, 'singleton solver'
    yield ac, solver1.composite.coefs, solver2.composite.coefs, 'container solver'

def test_quadratic_for_smooth():
    '''
    this test is a check to ensure that the
    quadratic part of the smooth functions are being used in the proximal step

    '''

    L = 0.45

    W = np.random.standard_normal(40)
    Z = np.random.standard_normal(40)
    U = np.random.standard_normal(40)

    atomq = rr.identity_quadratic(0.4, U, W, 0)
    atom = rr.l1norm(40, quadratic=atomq, lagrange=0.12)

    # specifying in this way should be the same as if we put 0.5*L below
    loss = rr.quadratic.shift(-Z, coef=0.6*L)
    lq = rr.identity_quadratic(0.4*L, Z, 0, 0)
    loss.quadratic = lq 

    ww = np.random.standard_normal(40)

    # specifying in this way should be the same as if we put 0.5*L below
    loss2 = rr.quadratic.shift(-Z, coef=L)
    np.testing.assert_allclose(loss2.objective(ww), loss.objective(ww))
    np.testing.assert_allclose(lq.objective(ww, 'func'), loss.nonsmooth_objective(ww))
    np.testing.assert_allclose(loss2.smooth_objective(ww, 'func'), 0.5 / 0.3 * loss.smooth_objective(ww, 'func'))
    np.testing.assert_allclose(loss2.smooth_objective(ww, 'grad'), 0.5 / 0.3 * loss.smooth_objective(ww, 'grad'))

    problem = rr.container(loss, atom)
    solver = rr.FISTA(problem)
    solver.fit(tol=1.0e-12)

    problem3 = rr.simple_problem(loss, atom)
    solver3 = rr.FISTA(problem3)
    solver3.fit(tol=1.0e-12, coef_stop=True)

    loss4 = rr.quadratic.shift(-Z, coef=0.6*L)
    problem4 = rr.simple_problem(loss4, atom)
    problem4.quadratic = lq
    solver4 = rr.FISTA(problem4)
    solver4.fit(tol=1.0e-12)

    gg_soln = rr.gengrad(problem4, L)

#     loss5 = rr.quadratic.shift(-Z, coef=0.6*L)
#     problem5 = rr.dual_problem(loss5.conjugate, rr.identity(loss5.primal_shape), atom.conjugate)
#     problem5.quadratic = lq
#     dsoln = problem5.solve()

    loss6 = rr.quadratic.shift(-Z, coef=0.6*L)
    loss6.quadratic = lq + atom.quadratic
    atomcp = copy(atom)
    atomcp.quadratic = rr.identity_quadratic(0,0,0,0)
    problem6 = rr.dual_problem(loss6.conjugate, rr.identity(loss6.primal_shape), atomcp.conjugate)
    problem6.lipschitz = L + atom.quadratic.coef
    dsoln2 = problem6.solve(coef_stop=True, tol=1.e-10, debug=True,
                            max_its=100)

    problem2 = rr.container(loss2, atom)
    solver2 = rr.FISTA(problem2)
    solver2.fit(tol=1.0e-12, coef_stop=True)

    q = rr.identity_quadratic(L, Z, 0, 0)

    ac(problem.objective(ww), atom.nonsmooth_objective(ww) + q.objective(ww,'func'))

    np.testing.assert_allclose(atom.solve(q), solver3.composite.coefs)
    np.testing.assert_allclose(atom.solve(q), gg_soln)
    np.testing.assert_allclose(atom.solve(q), solver2.composite.coefs)
    np.testing.assert_allclose(atom.solve(q), solver4.composite.coefs)
    # np.testing.assert_allclose(dsoln2, dsoln)
    np.testing.assert_allclose(atom.solve(q), dsoln2)
    np.testing.assert_allclose(atom.solve(q), solver.composite.coefs)

def test_quadratic_for_smooth2():
    '''
    this test is a check to ensure that the
    quadratic part of the smooth functions are being used in the proximal step

    '''

    L = 2

    W = np.arange(5)
    Z = 0.5 * np.arange(5)[::-1]
    U = 1.5 * np.arange(5)

    atomq = rr.identity_quadratic(0.4, U, W, 0)
    atom = rr.l1norm(5, quadratic=atomq, lagrange=0.1)

    # specifying in this way should be the same as if we put 0.5*L below
    loss = rr.quadratic.shift(-Z, coef=0.6*L)
    lq = rr.identity_quadratic(0.4*L, Z, 0, 0)
    loss.quadratic = lq 

    ww = np.ones(5)

    # specifying in this way should be the same as if we put 0.5*L below
    loss2 = rr.quadratic.shift(-Z, coef=L)
    np.testing.assert_allclose(loss2.objective(ww), loss.objective(ww))
    np.testing.assert_allclose(lq.objective(ww, 'func'), loss.nonsmooth_objective(ww))
    np.testing.assert_allclose(loss2.smooth_objective(ww, 'func'), 0.5 / 0.3 * loss.smooth_objective(ww, 'func'))
    np.testing.assert_allclose(loss2.smooth_objective(ww, 'grad'), 0.5 / 0.3 * loss.smooth_objective(ww, 'grad'))

    problem = rr.container(loss, atom)
    solver = rr.FISTA(problem)
    solver.fit(tol=1.0e-12)

    problem3 = rr.simple_problem(loss, atom)
    solver3 = rr.FISTA(problem3)
    solver3.fit(tol=1.0e-12, coef_stop=True)

    loss4 = rr.quadratic.shift(-Z, coef=0.6*L)
    problem4 = rr.simple_problem(loss4, atom)
    problem4.quadratic = lq
    solver4 = rr.FISTA(problem4)
    solver4.fit(tol=1.0e-12)

    gg_soln = rr.gengrad(problem4, L)

#     loss5 = rr.quadratic.shift(-Z, coef=0.6*L)
#     problem5 = rr.dual_problem(loss5.conjugate, rr.identity(loss5.primal_shape), atom.conjugate)
#     problem5.quadratic = lq
#     dsoln = problem5.solve()

    loss6 = rr.quadratic.shift(-Z, coef=0.6*L)
    loss6.quadratic = lq + atom.quadratic
    atomcp = copy(atom)
    atomcp.quadratic = rr.identity_quadratic(0,0,0,0)
    problem6 = rr.dual_problem(loss6.conjugate, rr.identity(loss6.primal_shape), atomcp.conjugate)
    problem6.lipschitz = L + atom.quadratic.coef
    dsoln2 = problem6.solve(coef_stop=True, tol=1.e-10, debug=True,
                            max_its=100, FISTA=False)

    problem2 = rr.container(loss2, atom)
    solver2 = rr.FISTA(problem2)
    solver2.fit(tol=1.0e-12, coef_stop=True)

    q = rr.identity_quadratic(L, Z, 0, 0)

    ac(problem.objective(ww), atom.nonsmooth_objective(ww) + q.objective(ww,'func'))

    np.testing.assert_allclose(atom.solve(q), solver3.composite.coefs)
    np.testing.assert_allclose(atom.solve(q), gg_soln)
    np.testing.assert_allclose(atom.solve(q), solver2.composite.coefs)
    np.testing.assert_allclose(atom.solve(q), solver4.composite.coefs)
    # np.testing.assert_allclose(dsoln2, dsoln)
    np.testing.assert_allclose(atom.solve(q), dsoln2)
    np.testing.assert_allclose(atom.solve(q), solver.composite.coefs)


