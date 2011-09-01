""" Testing D transform implementation
"""

from operator import add
import numpy as np
import regreg.api as rr
from regreg.affine import (broadcast_first, affine_transform, 
                           AffineError, composition, adjoint)

from numpy.testing import (assert_array_almost_equal,
                           assert_array_equal)


from nose.tools import assert_true, assert_equal, assert_raises


def test_broad_first():
    # Test broadcasting over second axis
    a = np.arange(4) + 10
    b = np.arange(4).reshape(4,1)
    c = broadcast_first(a, b, add)
    res = a[:,None] + b
    assert_equal(res.shape, c.shape)
    assert_array_equal(c, res)
    res1d = res.ravel()
    c = broadcast_first(b, a, add)
    assert_equal(res1d.shape, c.shape)
    assert_array_equal(c, res1d)
    c = broadcast_first(a, b.ravel(), add)
    assert_equal(res1d.shape, c.shape)
    assert_array_equal(c, res1d)


def test_affine_transform():
    # Test affine transform
    m = 20
    x1d = np.arange(m)
    x2d = x1d[:,None]
    x22d = np.c_[x2d, x2d]
    # Error if both of linear and affine components are None
    assert_raises(AffineError, affine_transform, None, None)
    # With linear None and 0 affine offset - identity transform
    for x in (x1d, x2d, x22d):
        trans = affine_transform(None, np.zeros((m,1)))
        assert_array_equal(trans.affine_map(x), x)
        assert_array_equal(trans.linear_map(x), x)
        assert_array_equal(trans.adjoint_map(x), x)
        # With linear eye and None affine offset - identity again
        trans = affine_transform(np.eye(m), None)
        assert_array_equal(trans.affine_map(x), x)
        assert_array_equal(trans.linear_map(x), x)
        assert_array_equal(trans.adjoint_map(x), x)
        # affine_transform as input
        trans = affine_transform(trans, None)
        assert_array_equal(trans.affine_map(x), x)
        assert_array_equal(trans.linear_map(x), x)
        assert_array_equal(trans.adjoint_map(x), x)
        # diag
        trans = affine_transform(np.ones(m), None, True)
        assert_array_equal(trans.affine_map(x), x)
        assert_array_equal(trans.linear_map(x), x)
        assert_array_equal(trans.adjoint_map(x), x)



def test_offset_transform():
    # Test affine transform
    m = 20
    x1d = np.arange(m)
    x2d = x1d[:,None]
    x22d = np.c_[x2d, x2d]
    # Error if both of linear and affine components are None
    assert_raises(AffineError, affine_transform, None, None)
    # With linear None and 0 affine offset - identity transform
    for x in (x1d, x2d, x22d):
        trans = affine_transform(None, np.zeros((m,1)))
        assert_array_equal(trans.affine_map(x), trans.offset_map(trans.linear_map(x)))
        # With linear eye and None affine offset - identity again
        trans = affine_transform(np.eye(m), None)
        assert_array_equal(trans.affine_map(x), trans.offset_map(trans.linear_map(x)))
        # affine_transform as input
        trans = affine_transform(trans, None)
        assert_array_equal(trans.affine_map(x), trans.offset_map(trans.linear_map(x)))
        # diag
        trans = affine_transform(np.ones(m), None, True)
        assert_array_equal(trans.affine_map(x), trans.offset_map(trans.linear_map(x)))
        #Non-zero offset
        trans = affine_transform(np.eye(m), np.ones((m,1)))
        assert_array_equal(trans.affine_map(x), trans.offset_map(trans.linear_map(x)))

def test_composition():
    X1 = np.random.standard_normal((20,30))
    X2 = np.random.standard_normal((30,10))
    b1 = np.random.standard_normal(20)
    b2 = np.random.standard_normal(30)
    L1 = affine_transform(X1, b1)
    L2 = affine_transform(X2, b2)

    z = np.random.standard_normal(10)
    w = np.random.standard_normal(20)
    comp = composition(L1,L2)

    assert_array_equal(comp.linear_map(z), np.dot(X1, np.dot(X2, z)))
    assert_array_equal(comp.adjoint_map(w), np.dot(X2.T, np.dot(X1.T, w)))
    assert_array_equal(comp.affine_map(z), np.dot(X1, np.dot(X2, z)+b2)+b1)

def test_composition2():
    X1 = np.random.standard_normal((20,30))
    X2 = np.random.standard_normal((30,10))
    X3 = np.random.standard_normal((10,20))

    b1 = np.random.standard_normal(20)
    b2 = np.random.standard_normal(30)
    b3 = np.random.standard_normal(10)

    L1 = affine_transform(X1, b1)
    L2 = affine_transform(X2, b2)
    L3 = affine_transform(X3, b3)

    z = np.random.standard_normal(20)
    w = np.random.standard_normal(20)
    comp = composition(L1,L2,L3)

    assert_array_equal(comp.linear_map(z), 
                       np.dot(X1, np.dot(X2, np.dot(X3, z))))
    assert_array_equal(comp.adjoint_map(w), 
                       np.dot(X3.T, np.dot(X2.T, np.dot(X1.T, w))))
    assert_array_equal(comp.affine_map(z), 
                       np.dot(X1, np.dot(X2, np.dot(X3, z) + b3) + b2) + b1)

def test_adjoint():
    X = np.random.standard_normal((20,30))
    b = np.random.standard_normal(20)
    L = affine_transform(X, b)

    z = np.random.standard_normal(30)
    w = np.random.standard_normal(20)
    A = adjoint(L)

    assert_array_equal(A.linear_map(w), L.adjoint_map(w))
    assert_array_equal(A.affine_map(w), L.adjoint_map(w))
    assert_array_equal(A.adjoint_map(z), L.linear_map(z))

def test_affine_sum():

    n = 100
    p = 25

    X1 = np.random.standard_normal((n,p))
    X2 = np.random.standard_normal((n,p))
    b = np.random.standard_normal(n)
    v = np.random.standard_normal(p)

    transform1 = rr.affine_transform(X1,b)
    transform2 = rr.linear_transform(X2)
    sum_transform = rr.affine_sum(transform1, transform2)

    assert_array_almost_equal(np.dot(X1,v) + np.dot(X2,v) + b, sum_transform.affine_map(v))
    assert_array_almost_equal(np.dot(X1,v) + np.dot(X2,v), sum_transform.linear_map(v))
    assert_array_almost_equal(np.dot(X1.T,b) + np.dot(X2.T,b), sum_transform.adjoint_map(b))
    assert_array_almost_equal(b, sum_transform.offset_map(v))
    assert_array_almost_equal(b, sum_transform.affine_offset)
