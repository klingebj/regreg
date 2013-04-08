import numpy as np, regreg.api as rr
import regreg.group_lasso as gl

def test_group_lasso_prox():
    prox_center = np.array([1,3,5,7,-9,3,4,6,7,8,9,11,13,4,-23,40], np.float)
    l1_penalty = np.array([0,1])
    unpenalized = np.array([2,3])
    positive_part = np.array([4,5])
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

    prox_result = gl.prox_group_lasso(prox_center, 1., 0.5, l1_penalty, unpenalized, positive_part, groups, weights)

    np.testing.assert_allclose(result, prox_result)

def test_group_lasso_atom():


    ps = np.array([0]*5 + [3]*3)
    weights = {3:2., 0:2.3}

    lagrange = 1.5
    lipschitz = 0.2
    p = gl.group_lasso(ps, lagrange, weights)
    z = 30 * np.random.standard_normal(8)
    q = rr.identity_quadratic(lipschitz, z, 0, 0)

    x = p.solve(q)
    a = gl.prox_group_lasso(z, lagrange, lipschitz, np.array([],np.int), np.array([],np.int), np.array([], np.int), np.array([0,0,0,0,0,1,1,1]), np.array([np.sqrt(5), 2]))

    result = np.zeros_like(a)
    result[:5] = z[:5] / np.linalg.norm(z[:5]) * max(np.linalg.norm(z[:5]) - weights[0] * lagrange/lipschitz, 0)
    result[5:] = z[5:] / np.linalg.norm(z[5:]) * max(np.linalg.norm(z[5:]) - weights[3] * lagrange/lipschitz, 0)

    lipschitz = 1.
    q = rr.identity_quadratic(lipschitz, z, 0, 0)
    x2 = p.solve(q)
    pc = p.conjugate
    a2 = pc.solve(q)

    np.testing.assert_allclose(z-a2, x2)

# penalty = rr.separable((100,), [rr.l2norm(idx[g].shape, lagrange=lagrange_g) for g in groups], groups)
# problem = rr.simple_problem(loss, penalty)

# from regreg.affine import tensor
# print coefs[groups[0]]
# penalty_block = rr.l1_l2((20,5), lagrange=0.0001)
# Xr = X.reshape((120,5,20)).transpose([0,2,1]) * 1.
# L = tensor(Xr, 1)
# loss_block = rr.squared_error(L,Y*1., coef=1./n)
# w = np.random.standard_normal(penalty_block.primal_shape)
# def test_grad():
#     w = np.random.standard_normal(penalty_block.primal_shape)
#     g_block = loss_block.smooth_objective(w, 'grad')
#     g = loss.smooth_objective(w.T.reshape(-1), 'grad')
#     # print sorted(g_block.reshape(-1))[:5]
#     # print sorted(g)[:5]
#     np.testing.assert_allclose(g[groups[1]], g_block.reshape((20,5))[1])
# print loss_block.primal_shape; test_grad()
# problem_block = rr.simple_problem(loss_block, penalty_block); test_grad()
# final_inv_step = lipschitz
# #block_coefs_gg = rr.gengrad(problem_block, lipschitz)
# block_coefs = problem_block.solve(tol=1.e-10, start_inv_step=final_inv_step);
# final_inv_step = problem_block.final_inv_step
# block_coefs = problem_block.solve(start_inv_step=final_inv_step, tol=1.e-14);
# final_inv_step = problem_block.final_inv_step
# block_coefs = problem_block.solve(start_inv_step=final_inv_step, coef_stop=True);

# test_grad()
# print problem_block.nonsmooth_objective(w), problem.nonsmooth_objective(w.T.reshape(-1)); test_grad()
# print problem_block.nonsmooth_objective(block_coefs), problem.nonsmooth_objective(coefs), 'huh'
# print problem_block.objective(block_coefs), problem.objective(coefs), 'huh1'
# print problem_block.objective(coefs.reshape((5,20)).T), problem.objective(block_coefs.T.reshape(-1)), 'huh2'
# print problem_block.objective(coefs.reshape((5,20)).T), problem.objective(block_coefs.T.reshape(-1)), 'huh3'
# print problem_block.objective(block_coefs), problem.objective(block_coefs.T.reshape(-1)), 'huhblock'
# print problem_block.objective(coefs.reshape((5,20)).T), problem.objective(coefs), 'huhcoefs'
# print 'prox: 3', np.linalg.norm(penalty_block.lagrange_prox(w, lipschitz=3, lagrange=0.5)[0] - penalty.atoms[0].lagrange_prox(w.T.reshape(-1)[groups[0]], lagrange=0.5, lipschitz=3))
# print 'prox: 1', np.linalg.norm(penalty_block.lagrange_prox(w, lipschitz=1, lagrange=0.5)[0] - penalty.atoms[0].lagrange_prox(w.T.reshape(-1)[groups[0]], lagrange=0.5, lipschitz=1))
# print 'prox all', np.linalg.norm(penalty_block.proximal(rr.identity_quadratic(3, w, 0, 0)) -
#                                  penalty.proximal(rr.identity_quadratic(3,w,0,0)).reshape((5,20)).T)
# print 'indexing: ', w[0], w.T.reshape(-1)[groups[0]]; test_grad()
# a=penalty.atoms[0]; v=w[0]; test_grad()
# print 'atom:', a; test_grad()
# print 'atom prox: ', a.lagrange_prox(w[0], lipschitz=1, lagrange=0.5); test_grad()
# nn = np.linalg.norm(w[0]); test_grad()


# print 'by hand: ', ((nn-0.5)/nn)*w[0]; test_grad()
# print block_coefs.T.reshape(-1)[groups[0]], coefs[groups[0]]
# problem = rr.simple_problem(loss, penalty)
# #coefs = problem.solve(); 
# #coefs = problem.coefs.copy()
# print sorted(block_coefs.reshape(-1))[:5], sorted(coefs)[:5], 'sorted'
# print 'agreement: ', np.linalg.norm(coefs-block_coefs.T.reshape(-1)) / np.linalg.norm(coefs)
