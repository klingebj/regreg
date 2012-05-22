
import numpy as np

from copy import copy
import scipy.optimize

import regreg.api as rr
from test_seminorms import ac

def test_quadratic():

    l = rr.quadratic(5, coef=3., offset=np.arange(5))
    l.quadratic = rr.identity_quadratic(1,np.ones(5), 2*np.ones(5), 3.)
    c1 = l.conjugate

    q1 = rr.identity_quadratic(3, -np.arange(5), 0, 0)
    q2 = q1 + l.quadratic
    c2 = rr.zero(5, quadratic=q2.collapsed()).conjugate

    ww = np.random.standard_normal(5)
    np.testing.assert_almost_equal(c2.smooth_objective(ww, 'grad'),
                                   c1.smooth_objective(ww, 'grad'))

    np.testing.assert_almost_equal(c2.objective(ww),
                                   c1.objective(ww))

    np.testing.assert_almost_equal(c2.smooth_objective(ww, 'func'),
                                   c1.smooth_objective(ww, 'func'))

    np.testing.assert_almost_equal(c2.nonsmooth_objective(ww),
                                   c1.nonsmooth_objective(ww))
