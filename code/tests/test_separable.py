import numpy as np
import regreg.api as rr
import nose.tools as nt

def test_lasso_separable():
    """
    This test verifies that the specification of a separable
    penalty yields the same results as having two linear_atoms
    with selector matrices. The penalty here is a lasso, i.e. l1
    penalty.
    """

    X = np.random.standard_normal((100,20))
    Y = np.random.standard_normal((100,)) + np.dot(X, np.random.standard_normal(20))

    penalty1 = rr.l1norm(10, lagrange=.2)
    penalty2 = rr.l1norm(10, lagrange=.2)
    penalty = rr.separable((20,), [penalty1, penalty2], [slice(0,10), slice(10,20)])

    # solve using separable
    
    loss = rr.l2normsq.affine(X, -Y, coef=0.5)
    problem = rr.container(loss, penalty)
    solver = rr.FISTA(problem)
    solver.fit(min_its=200, tol=1.0e-12)
    coefs = solver.composite.coefs

    # solve using the usual composite

    penalty_all = rr.l1norm(20, lagrange=.2)
    problem_all = rr.container(loss, penalty_all)
    solver_all = rr.FISTA(problem_all)
    solver_all.fit(min_its=200, tol=1.0e-12)

    coefs_all = solver_all.composite.coefs

    # solve using the selectors

    penalty_s = [rr.linear_atom(p, rr.selector(g, (20,))) for p, g in
                 zip(penalty.atoms, penalty.groups)]
    problem_s = rr.container(loss, *penalty_s)
    solver_s = rr.FISTA(problem_s)
    solver_s.fit(min_its=200, tol=1.0e-12)
    coefs_s = solver_s.composite.coefs

    np.testing.assert_almost_equal(coefs, coefs_all)
    np.testing.assert_almost_equal(coefs, coefs_s)


def test_group_lasso_separable():
    """
    This test verifies that the specification of a separable
    penalty yields the same results as having two linear_atoms
    with selector matrices. The penalty here is a group_lasso, i.e. l2
    penalty.
    """

    X = np.random.standard_normal((100,20))
    Y = np.random.standard_normal((100,)) + np.dot(X, np.random.standard_normal(20))

    penalty1 = rr.l2norm(10, lagrange=.2)
    penalty2 = rr.l2norm(10, lagrange=.2)
    penalty = rr.separable((20,), [penalty1, penalty2], [slice(0,10), slice(10,20)])

    # solve using separable
    
    loss = rr.l2normsq.affine(X, -Y, coef=0.5)
    problem = rr.container(loss, penalty)
    solver = rr.FISTA(problem)
    solver.fit(min_its=200, tol=1.0e-12)
    coefs = solver.composite.coefs

    # solve using the selectors

    penalty_s = [rr.linear_atom(p, rr.selector(g, (20,))) for p, g in
                 zip(penalty.atoms, penalty.groups)]
    problem_s = rr.container(loss, *penalty_s)
    solver_s = rr.FISTA(problem_s)
    solver_s.fit(min_its=200, tol=1.0e-12)
    coefs_s = solver_s.composite.coefs

    np.testing.assert_almost_equal(coefs, coefs_s)

def test_nonnegative_positive_part(debug=False):
    """
    This test verifies that using nonnegative constraint
    with a linear term, with some unpenalized terms yields the same result
    as using separable with constrained_positive_part and nonnegative
    """
    import numpy as np
    import regreg.api as rr
    import regreg.atoms as rra

    # N - number of data points
    # P - number of columns in design == number of betas
    N, P = 40, 30
    # an arbitrary positive offset for data and design
    offset = 2
    # data
    Y = np.random.normal(size=(N,)) + offset
    # design - with ones as last column
    X = np.ones((N,P))
    X[:,:-1] = np.random.normal(size=(N,P-1)) + offset
    # coef for loss
    coef = 0.5
    # lagrange for penalty
    lagrange = .1

    # Loss function (squared difference between fitted and actual data)
    loss = rr.l2normsq.affine(X, -Y, coef=coef)

    # Penalty using nonnegative, leave the last 5 unpenalized but
    # nonnegative
    weights = np.ones(P) * lagrange
    weights[-5:] = 0
    penalty = rr.nonnegative(P, linear_term=weights)

    # Solution
    problem = rr.container(loss, penalty)
    composite_form = rr.composite(loss.smooth_objective,
                                  penalty.nonsmooth_objective, penalty.proximal, np.random.standard_normal(P))
    solver = rr.FISTA(composite_form)
    solver.debug = debug
    solver.fit(tol=1.0e-12, min_its=200)
    coefs = solver.composite.coefs

    # using the separable penalty, only penalize the first
    # 25 coefficients with constrained_positive_part

    penalties_s = [rr.constrained_positive_part(25, lagrange=lagrange),
                   rr.nonnegative(5)]
    groups_s = [slice(0,25), slice(25,30)]
    penalty_s = rr.separable((P,), penalties_s,
                             groups_s)
    composite_form_s = rr.composite(loss.smooth_objective,
                                    penalty_s.nonsmooth_objective,
                                    penalty_s.proximal,
                                    np.random.standard_normal(P))
    solver_s = rr.FISTA(composite_form_s)
    solver_s.debug = debug
    solver_s.fit(tol=1.0e-12, min_its=200)
    coefs_s = solver_s.composite.coefs

    nt.assert_true(np.linalg.norm(coefs - coefs_s) / np.linalg.norm(coefs) < 1.0e-02)

