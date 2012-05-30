penalty = rr.separable((100,), [rr.l2norm(idx[g].shape, lagrange=lagrange_g) for g in groups], groups)
problem = rr.simple_problem(loss, penalty)

from regreg.affine import tensor
print coefs[groups[0]]
penalty_block = rr.l1_l2((20,5), lagrange=0.0001)
Xr = X.reshape((120,5,20)).transpose([0,2,1]) * 1.
L = tensor(Xr, 1)
loss_block = rr.squared_error(L,Y*1., coef=1./n)
w = np.random.standard_normal(penalty_block.primal_shape)
def test_grad():
    w = np.random.standard_normal(penalty_block.primal_shape)
    g_block = loss_block.smooth_objective(w, 'grad')
    g = loss.smooth_objective(w.T.reshape(-1), 'grad')
    # print sorted(g_block.reshape(-1))[:5]
    # print sorted(g)[:5]
    np.testing.assert_allclose(g[groups[1]], g_block.reshape((20,5))[1])
print loss_block.primal_shape; test_grad()
problem_block = rr.simple_problem(loss_block, penalty_block); test_grad()
final_inv_step = lipschitz
#block_coefs_gg = rr.gengrad(problem_block, lipschitz)
block_coefs = problem_block.solve(tol=1.e-10, start_inv_step=final_inv_step);
final_inv_step = problem_block.final_inv_step
block_coefs = problem_block.solve(start_inv_step=final_inv_step, tol=1.e-14);
final_inv_step = problem_block.final_inv_step
block_coefs = problem_block.solve(start_inv_step=final_inv_step, coef_stop=True);

test_grad()
print problem_block.nonsmooth_objective(w), problem.nonsmooth_objective(w.T.reshape(-1)); test_grad()
print problem_block.nonsmooth_objective(block_coefs), problem.nonsmooth_objective(coefs), 'huh'
print problem_block.objective(block_coefs), problem.objective(coefs), 'huh1'
print problem_block.objective(coefs.reshape((5,20)).T), problem.objective(block_coefs.T.reshape(-1)), 'huh2'
print problem_block.objective(coefs.reshape((5,20)).T), problem.objective(block_coefs.T.reshape(-1)), 'huh3'
print problem_block.objective(block_coefs), problem.objective(block_coefs.T.reshape(-1)), 'huhblock'
print problem_block.objective(coefs.reshape((5,20)).T), problem.objective(coefs), 'huhcoefs'
print 'prox: 3', np.linalg.norm(penalty_block.lagrange_prox(w, lipschitz=3, lagrange=0.5)[0] - penalty.atoms[0].lagrange_prox(w.T.reshape(-1)[groups[0]], lagrange=0.5, lipschitz=3))
print 'prox: 1', np.linalg.norm(penalty_block.lagrange_prox(w, lipschitz=1, lagrange=0.5)[0] - penalty.atoms[0].lagrange_prox(w.T.reshape(-1)[groups[0]], lagrange=0.5, lipschitz=1))
print 'prox all', np.linalg.norm(penalty_block.proximal(rr.identity_quadratic(3, w, 0, 0)) -
                                 penalty.proximal(rr.identity_quadratic(3,w,0,0)).reshape((5,20)).T)
print 'indexing: ', w[0], w.T.reshape(-1)[groups[0]]; test_grad()
a=penalty.atoms[0]; v=w[0]; test_grad()
print 'atom:', a; test_grad()
print 'atom prox: ', a.lagrange_prox(w[0], lipschitz=1, lagrange=0.5); test_grad()
nn = np.linalg.norm(w[0]); test_grad()


print 'by hand: ', ((nn-0.5)/nn)*w[0]; test_grad()
print block_coefs.T.reshape(-1)[groups[0]], coefs[groups[0]]
problem = rr.simple_problem(loss, penalty)
#coefs = problem.solve(); 
#coefs = problem.coefs.copy()
print sorted(block_coefs.reshape(-1))[:5], sorted(coefs)[:5], 'sorted'
print 'agreement: ', np.linalg.norm(coefs-block_coefs.T.reshape(-1)) / np.linalg.norm(coefs)
