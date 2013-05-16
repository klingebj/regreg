import numpy as np, regreg.api as rr
import regreg.atoms.mixed_lasso as ml
import regreg.atoms.group_lasso as gl

def test_group_lasso_prox():
    prox_center = np.array([1,3,5,7,-9,3,4,6,7,8,9,11,13,4,-23,40], np.float)
    l1_penalty = np.array([0,1], np.int)
    unpenalized = np.array([2,3], np.int)
    positive_part = np.array([4,5], np.int)
    nonnegative = np.array([], np.int)
    groups = np.array([-2,-2,-1,-1,-3,-3] + [0]*5 + [1]*5)
    weights = np.array([1.,1.])

    lagrange = 1.
    lipschitz = 0.5

    result = np.zeros_like(prox_center)

    result[unpenalized] = prox_center[unpenalized]
    result[positive_part] = (prox_center[positive_part] - lagrange / lipschitz) * np.maximum(prox_center[positive_part] - lagrange / lipschitz, 0)
    result[l1_penalty] = np.maximum(np.fabs(prox_center[l1_penalty]) - lagrange / lipschitz, 0) * np.sign(prox_center[l1_penalty])

    result[6:11] = prox_center[6:11] / np.linalg.norm(prox_center[6:11]) * max(np.linalg.norm(prox_center[6:11]) - weights[0] * lagrange/lipschitz, 0)
    result[11:] = prox_center[11:] / np.linalg.norm(prox_center[11:]) * max(np.linalg.norm(prox_center[11:]) - weights[1] * lagrange/lipschitz, 0)

    prox_result = ml.mixed_lasso_lagrange_prox(prox_center, lagrange, lipschitz, l1_penalty, unpenalized, positive_part, nonnegative, groups, weights)

    np.testing.assert_allclose(result, prox_result)

def test_group_lasso_atom():


    ps = np.array([0]*5 + [3]*3)
    weights = {3:2., 0:2.3}

    lagrange = 1.5
    lipschitz = 0.2
    p = gl.group_lasso(ps, weights=weights, lagrange=lagrange)
    z = 30 * np.random.standard_normal(8)
    q = rr.identity_quadratic(lipschitz, z, 0, 0)

    x = p.solve(q)
    a = ml.mixed_lasso_lagrange_prox(z, lagrange, lipschitz, 
                                     np.array([],np.int), 
                                     np.array([],np.int), 
                                     np.array([], np.int), 
                                     np.array([], np.int), 
                                     np.array([0,0,0,0,0,1,1,1]), np.array([np.sqrt(5), 2]))

    result = np.zeros_like(a)
    result[:5] = z[:5] / np.linalg.norm(z[:5]) * max(np.linalg.norm(z[:5]) - weights[0] * lagrange/lipschitz, 0)
    result[5:] = z[5:] / np.linalg.norm(z[5:]) * max(np.linalg.norm(z[5:]) - weights[3] * lagrange/lipschitz, 0)

    lipschitz = 1.
    q = rr.identity_quadratic(lipschitz, z, 0, 0)
    x2 = p.solve(q)
    pc = p.conjugate
    a2 = pc.solve(q)

    np.testing.assert_allclose(z-a2, x2)

