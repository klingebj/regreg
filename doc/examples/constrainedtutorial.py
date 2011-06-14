import numpy as np
import pylab	
from scipy import sparse

import regreg.api as rr

Y = np.random.standard_normal(500); Y[100:150] += 7; Y[250:300] += 14
loss = rr.l2normsq.shift(-Y, coef=0.5)

sparsity = rr.l1norm(len(Y), 1.4)
# TODO should make a module to compute typical Ds
D = sparse.csr_matrix((np.identity(500) + np.diag([-1]*499,k=1))[:-1])
fused = rr.l1norm.linear(D, 25.5)
problem = rr.container(loss, sparsity, fused)

solver = rr.FISTA(problem)
solver.fit(max_its=100, tol=1e-10)
solution = solver.composite.coefs

delta1 = np.fabs(D * solution).sum()
delta2 = np.fabs(solution).sum()

fused_constraint = rr.l1norm.linear(D, bound=delta1)
sparsity_constraint = rr.l1norm(500, bound=delta2)

constrained_problem = rr.container(loss, fused_constraint, sparsity_constraint)
constrained_solver = rr.FISTA(constrained_problem)
constrained_solver.composite.lipschitz = 1.01
vals = constrained_solver.fit(max_its=10, tol=1e-06, backtrack=False, monotonicity_restart=False)
constrained_solution = constrained_solver.composite.coefs

loss = rr.l2normsq.shift(-Y, coef=0.5)
true_conjugate = rr.l2normsq.shift(Y, coef=0.5, constant_term=-np.linalg.norm(Y)**2/2)
problem = rr.container(loss, fused_constraint, sparsity_constraint)
solver = rr.FISTA(problem.conjugate_composite(true_conjugate))
solver.fit(max_its=200, tol=1e-08)
conjugate_coefs = problem.conjugate_primal_from_dual(solver.composite.coefs)

loss = rr.l2normsq.shift(-Y, coef=0.5)
problem = rr.container(loss, fused_constraint, sparsity_constraint)
solver = rr.FISTA(problem.conjugate_composite())
solver.fit(max_its=200, tol=1e-08)
conjugate_coefs_gen = problem.conjugate_primal_from_dual(solver.composite.coefs)

pylab.scatter(np.arange(Y.shape[0]), Y)

pylab.plot(solution, c='y', linewidth=7)	
pylab.plot(constrained_solution, c='r', linewidth=5)
pylab.plot(conjugate_coefs, c='black', linewidth=3)	
pylab.plot(conjugate_coefs_gen, c='gray', linewidth=1)		
