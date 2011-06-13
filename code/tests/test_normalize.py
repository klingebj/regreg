import numpy as np
import regreg.api as rr
import nose.tools as nt

def test_centering():
    """
    This test verifies that the normalized transform
    of affine correctly implements the linear
    transform that multiplies first by X, then centers.
    """
    # N - number of data points
    # P - number of columns in design == number of betas
    N, P = 40, 30
    # an arbitrary positive offset for data and design
    offset = 50

    # design - with ones as last column
    X = np.ones((N,P))
    X[:,:-1] = np.random.normal(size=(N,P-1)) + offset
    X2 = X - X.mean(axis=0)[np.newaxis,:]
    L = rr.normalize(X, center=True, scale=False)
    # coef for loss

    beta = np.random.normal(size=(P,))
    v = L.linear_map(beta)
    v2 = np.dot(X, beta)
    v2 -= v2.mean()
    v3 = np.dot(X2, beta)
    np.testing.assert_almost_equal(v, v2)
    np.testing.assert_almost_equal(v, v3)

def test_scaling():
    """
    This test verifies that the normalized transform
    of affine correctly implements the linear
    transform that multiplies first by X, then centers.
    """
    # N - number of data points
    # P - number of columns in design == number of betas
    N, P = 40, 30
    # an arbitrary positive offset for data and design
    offset = 50

    # design - with ones as last column
    X = np.ones((N,P))
    X[:,:-1] = np.random.normal(size=(N,P-1)) + offset

    L = rr.normalize(X, center=False, scale=True)
    # coef for loss

    scalings = np.sqrt((X**2).sum(0) / N)
    scaling_matrix = np.diag(1./scalings)
    beta = np.random.normal(size=(P,))
    v = L.linear_map(beta)
    v2 = np.dot(X, np.dot(scaling_matrix, beta))
    np.testing.assert_almost_equal(v, v2)

def test_scaling_and_centering():
    """
    This test verifies that the normalized transform
    of affine correctly implements the linear
    transform that multiplies first by X, then centers.
    """
    # N - number of data points
    # P - number of columns in design == number of betas
    N, P = 40, 30
    # an arbitrary positive offset for data and design
    offset = 50

    # design - with no colum of ones!
    X = np.random.normal(size=(N,P)) + offset

    L = rr.normalize(X, center=True, scale=True) # the default
    # coef for loss

    scalings = np.std(X, 0)
    scaling_matrix = np.diag(1./scalings)
    beta = np.random.normal(size=(P,))
    v = L.linear_map(beta)
    v2 = np.dot(X, np.dot(scaling_matrix, beta))
    v2 -= v2.mean()
    np.testing.assert_almost_equal(v, v2)


@np.testing.decorators.knownfailureif(True, msg='the intercept coefficient is off here')
def test_centering_fit(debug=False):

    # N - number of data points
    # P - number of columns in design == number of betas
    N, P = 40, 30
    # an arbitrary positive offset for data and design
    offset = 50

    # design - with ones as last column
    X = np.ones((N,P))
    X[:,:-1] = np.random.normal(size=(N,P-1)) + offset
    X2 = X - X.mean(axis=0)[np.newaxis,:]

    # the normalizer
    L = rr.normalize(X, center=True, scale=False)

    # data
    Y = np.random.normal(size=(N,)) + offset

    # coef for loss
    coef = 0.5
    # lagrange for penalty
    lagrange = .1

    # Loss function (squared difference between fitted and actual data)
    loss = rr.l2normsq.affine(L, -Y, coef=coef)

    penalties = [rr.constrained_positive_part(25, lagrange=lagrange),
                 rr.nonnegative(5, lagrange=1)]
    groups = [slice(0,25), slice(25,30)]
    penalty = rr.separable((P,), penalties,
                           groups)
    composite_form = rr.composite(loss.smooth_objective,
                                  penalty.nonsmooth_objective,
                                  penalty.proximal,
                                  np.random.standard_normal(P))
    solver = rr.FISTA(composite_form)
    solver.debug = debug
    solver.fit(tol=1.0e-12, min_its=200)
    coefs = solver.composite.coefs

    # Solve the problem with X2
    loss2 = rr.l2normsq.affine(X2, -Y, coef=coef)

    composite_form2 = rr.composite(loss2.smooth_objective,
                                  penalty.nonsmooth_objective,
                                  penalty.proximal,
                                  np.random.standard_normal(P))
    solver2 = rr.FISTA(composite_form2)
    solver2.debug = debug
    solver2.fit(tol=1.0e-12, min_its=200)
    coefs2 = solver2.composite.coefs

    print np.linalg.norm(coefs - coefs2) / np.linalg.norm(coefs)
    nt.assert_true(np.linalg.norm(coefs - coefs2) / max(np.linalg.norm(coefs),1) < 1.0e-04)

@np.testing.decorators.knownfailureif(True, msg='the scaling is off, even though the previous tests pass')
def test_scaling_fit(debug=False):

    # N - number of data points
    # P - number of columns in design == number of betas
    N, P = 40, 30
    # an arbitrary positive offset for data and design
    offset = 2

    # design - with ones as last column
    X = np.ones((N,P))
    X[:,:-1] = np.random.normal(size=(N,P-1)) + offset
    X2 = X / np.sqrt((X**2).sum(0) / N)[np.newaxis,0]

    L = rr.normalize(X, center=False, scale=True)
    # coef for loss

    # data
    Y = np.random.normal(size=(N,)) + offset

    # coef for loss
    coef = 0.5
    # lagrange for penalty
    lagrange = .1

    # Loss function (squared difference between fitted and actual data)
    loss = rr.l2normsq.affine(L, -Y, coef=coef)

    penalties = [rr.constrained_positive_part(25, lagrange=lagrange),
                 rr.nonnegative(5, lagrange=1)]
    groups = [slice(0,25), slice(25,30)]
    penalty = rr.separable((P,), penalties,
                           groups)
    composite_form = rr.composite(loss.smooth_objective,
                                  penalty.nonsmooth_objective,
                                  penalty.proximal,
                                  np.random.standard_normal(P))
    solver = rr.FISTA(composite_form)
    solver.debug = debug
    solver.fit(tol=1.0e-12, min_its=200)
    coefs = solver.composite.coefs

    # Solve the problem with X2
    loss2 = rr.l2normsq.affine(X2, -Y, coef=coef)

    composite_form2 = rr.composite(loss2.smooth_objective,
                                  penalty.nonsmooth_objective,
                                  penalty.proximal,
                                  np.random.standard_normal(P))
    solver2 = rr.FISTA(composite_form2)
    solver2.debug = debug
    solver2.fit(tol=1.0e-12, min_its=200)
    coefs2 = solver2.composite.coefs

    print np.linalg.norm(coefs - coefs2) / np.linalg.norm(coefs)
    nt.assert_true(np.linalg.norm(coefs - coefs2) / max(np.linalg.norm(coefs),1) < 1.0e-04)

@np.testing.decorators.knownfailureif(True, msg='the scaling and centering are off, even though the previous tests pass')
def test_scaling_and_centering_fit(debug=False):

    # N - number of data points
    # P - number of columns in design == number of betas
    N, P = 40, 30
    # an arbitrary positive offset for data and design
    offset = 2

    # design - with ones as last column
    X = np.random.normal(size=(N,P)) + offset
    X2 = X - X.mean(0)[np.newaxis,0]
    X2 = X2 / np.std(X2)

    L = rr.normalize(X, center=True, scale=True)
    # data
    Y = np.random.normal(size=(N,)) + offset

    # coef for loss
    coef = 0.5
    # lagrange for penalty
    lagrange = .1

    # Loss function (squared difference between fitted and actual data)
    loss = rr.l2normsq.affine(L, -Y, coef=coef)

    penalties = [rr.constrained_positive_part(25, lagrange=lagrange),
                 rr.nonnegative(5, lagrange=1)]
    groups = [slice(0,25), slice(25,30)]
    penalty = rr.separable((P,), penalties,
                           groups)
    composite_form = rr.composite(loss.smooth_objective,
                                  penalty.nonsmooth_objective,
                                  penalty.proximal,
                                  np.random.standard_normal(P))
    solver = rr.FISTA(composite_form)
    solver.debug = debug
    solver.fit(tol=1.0e-12, min_its=200)
    coefs = solver.composite.coefs

    # Solve the problem with X2
    loss2 = rr.l2normsq.affine(X2, -Y, coef=coef)

    composite_form2 = rr.composite(loss2.smooth_objective,
                                  penalty.nonsmooth_objective,
                                  penalty.proximal,
                                  np.random.standard_normal(P))
    solver2 = rr.FISTA(composite_form2)
    solver2.debug = debug
    solver2.fit(tol=1.0e-12, min_its=200)
    coefs2 = solver2.composite.coefs

    print np.linalg.norm(coefs - coefs2) / np.linalg.norm(coefs)
    nt.assert_true(np.linalg.norm(coefs - coefs2) / max(np.linalg.norm(coefs),1) < 1.0e-04)

