import numpy as np
import regreg.api as rr

def test_conjugate_l1norm():
    '''
    this test verifies that numerically computing the conjugate
    is essentially the same as using the smooth_conjugate
    of the atom
    '''


    l1 = rr.l1norm(4, lagrange=0.3)
    v1=rr.smooth_conjugate(l1, rr.identity_quadratic(0.3,None,None,0))
    v2 = rr.conjugate(l1, rr.identity_quadratic(0.3,None,None,0), tol=1.e-12)
    w=np.random.standard_normal(4)

    u11, u12 = v1.smooth_objective(w)
    u21, u22 = v2.smooth_objective(w)
    np.testing.assert_approx_equal(u11, u21)
    np.testing.assert_allclose(u12, u22, rtol=1.0e-05)

def test_conjugate_sqerror():

    X = np.random.standard_normal((10,4))
    Y = np.random.standard_normal(10)
    l = rr.quadratic.affine(X,-Y, coef=0.5)
    v = rr.conjugate(l, rr.identity_quadratic(0.3,None,None,0), tol=1.e-12)
    w=np.random.standard_normal(4)
    u11, u12 = v.smooth_objective(w)

    XTX = np.dot(X.T, X) 
    b = u22 = np.linalg.solve(XTX + 0.3 * np.identity(4), np.dot(X.T, Y) + w)
    u21 = - np.dot(b.T, np.dot(XTX + 0.3 * np.identity(4), b)) / 2. + (w*b).sum()  + (np.dot(X.T, Y) * b).sum() - np.linalg.norm(Y)**2/2.
    np.testing.assert_approx_equal(u11, u21)
    np.testing.assert_allclose(u12, u22, rtol=1.0e-05)
    
