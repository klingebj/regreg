import numpy as np
from scipy import sparse

from .affine import astransform, affine_transform


def difference_transform(X, order=1, sorted=False,
                         transform=False):
    """
    Compute the divided difference matrix for X
    after sorting X.

    Parameters
    ----------

    X: np.array, np.float, ndim=1

    order: int
        What order of difference should we compute?

    sorted: bool
        Is X sorted?

    transform: bool
        If True, return a linear_transform rather
        than a sparse matrix.

    Returns
    -------

    D: np.array, ndim=2, shape=(n-order,order)
        Matrix of divided differences of sorted X.

    """
    if not sorted:
        X = np.sort(X)
    X = np.asarray(X)
    n = X.shape[0]
    Dfinal = np.identity(n)
    for j in range(1, order+1):
        D = (-np.identity(n-j+1)+np.diag(np.ones(n-j),k=1))[:-1]
        steps = X[j:]-X[:-j]
        inv_steps = np.zeros(steps.shape)
        inv_steps[steps != 0] = 1. / steps[steps != 0]
        D = np.dot(np.diag(inv_steps), D)
        Dfinal = np.dot(D, Dfinal)
    if not transform:
        return sparse.csr_matrix(Dfinal)
    return astransform(Dfinal)

class trend_filter(affine_transform):

    def __init__(self, m, order=1, knots=None, sorted=False):
        self.m = m
        self.order = order
        if knots is None:
            knots = np.arange(m)
        else:
            knots = np.sort(knots)

        self.knots = knots
        self.steps = knots[1:] - knots[:-1]

        self.linear_transform = difference_transform(knots, order=order, sorted=True,
                                                     transform=True)
        self.affine_offset = None
        self.input_shape = self.linear_transform.input_shape
        self.output_shape = self.linear_transform.output_shape

    def linear_map(self, x):
        return self.linear_transform.linear_map(x)

    def affine_map(self, x):
        return self.linear_map(x)

    def offset_map(self, x):
        return x

    def adjoint_map(self, x):
        return self.linear_transform.adjoint_map(x)


class trend_filter_inverse(affine_transform):

    def __init__(self, m, order=1, knots=None, sorted=False):
        self.m = m
        self.order = order
        if order != 1:
            raise ValueError('pinv only worked out for first order')
        if knots is None:
            knots = np.arange(m)
        else:
            knots = np.sort(knots)

        self.knots = knots
        self.steps = knots[1:] - knots[:-1]
            
        dtransform = difference_transform(knots, order=order, sorted=True,
                                          transform=True)

        self.affine_offset = None
        self.output_shape = dtransform.input_shape
        self.input_shape = dtransform.output_shape

    def linear_map(self, x):
        if x.ndim == 1:
            v = np.zeros(self.m)
            v[1:] = np.cumsum(x * self.steps)
            v -= v.mean()
            return v
        elif x.ndim == 2:
            # assuming m is the first axis
            v = np.zeros((self.m, x.shape[1]))
            v[1:] = np.cumsum(x * self.steps[:,np.newaxis], axis=0)
            v -= v.mean(0)
            return v

    def affine_map(self, x):
        return self.linear_map(x)

    def offset_map(self, x):
        return x

    def adjoint_map(self, x):
        if x.ndim == 1:
            x = x - x.mean(0)
            C = np.cumsum(x[1:][::-1])[::-1]
            return C * self.steps
        if x.ndim == 2:
            # assuming m is the first axis
            x = x - x.mean(0)[np.newaxis,:]
            C = np.cumsum(x[1:][::-1], 1)[::-1]
            return C * self.steps[:,np.newaxis]


