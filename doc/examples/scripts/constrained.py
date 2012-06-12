import numpy as np
import pylab	
from scipy import sparse

from regreg.algorithms import FISTA
from regreg.atoms import l1norm
from regreg.container import container
from regreg.smooth import l2normsq
 
Y = np.random.standard_normal(500); Y[100:150] += 7; Y[250:300] += 14
loss = l2normsq.shift(-Y, coef=0.5)

sparsity = l1norm(len(Y), 1.4)
# TODO should make a module to compute typical Ds
D = sparse.csr_matrix((np.identity(500) + np.diag([-1]*499,k=1))[:-1])
fused = l1norm.linear(D, 25.5)
problem = container(loss, sparsity, fused)
   
solver = FISTA(problem.composite())
solver.fit(max_its=100, tol=1e-10)
solution = solver.composite.coefs

delta1 = np.fabs(D * solution).sum()
delta2 = np.fabs(solution).sum()

fused_constraint = l1norm.linear(D, bound=delta1)
sparsity_constraint = l1norm(500, bound=delta2)

constrained_problem = container(loss, fused_constraint, sparsity_constraint)
constrained_solver = FISTA(constrained_problem.composite())
constrained_solver.composite.lipshitz = 1.01
vals = constrained_solver.fit(max_its=10, tol=1e-06, backtrack=False, monotonicity_restart=False)
constrained_solution = constrained_solver.composite.coefs



loss = l2normsq.shift(-Y, coef=0.5)
true_conjugate = l2normsq.shift(Y, coef=0.5)
problem = container(loss, fused_constraint, sparsity_constraint)
solver = FISTA(problem.conjugate_composite(true_conjugate))
solver.fit(max_its=200, tol=1e-08)
conjugate_coefs = problem.conjugate_primal_from_dual(solver.composite.coefs)

from regreg.conjugate import conjugate

loss = l2normsq.shift(-Y, coef=0.5)
problem = container(loss, fused_constraint, sparsity_constraint)
solver = FISTA(problem.conjugate_composite())
solver.fit(max_its=200, tol=1e-08)
conjugate_coefs_gen = problem.conjugate_primal_from_dual(solver.composite.coefs)



pylab.scatter(np.arange(Y.shape[0]), Y)

pylab.plot(solution, c='y', linewidth=7)	
pylab.plot(constrained_solution, c='r', linewidth=5)
pylab.plot(conjugate_coefs, c='black', linewidth=3)	
pylab.plot(conjugate_coefs_gen, c='gray', linewidth=1)		
