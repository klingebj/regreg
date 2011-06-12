""" Testing D transform implementation
"""

from operator import add
import numpy as np

from regreg.affine import broadcast_first, affine_transform, AffineError

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
