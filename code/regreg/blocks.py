import numpy as np
import seminorm, smooth, algorithms
import pylab

def dual_blocks(semi, Y, initial=None):
    problems = []
    solvers = []
    losses = []
    if initial is None:
        initial = [None] * len(semi.atoms)
    for atom, atom_initial in zip(semi.atoms, initial):
        if atom.D is not None:
            loss = smooth.squaredloss(atom.D.T, Y.copy())
        else:
            loss = smooth.signal_approximator(Y.copy())
        losses.append(loss)

        dual_atom = atom.dual_constraint
        prox = dual_atom.primal_prox
        nonsmooth = dual_atom.evaluate
        if atom_initial is None:
            atom_initial = np.zeros(dual_atom.p) 
        if dual_atom.evaluate(atom_initial) == np.inf:
            raise ValueError('initial point is not feasible')
        
        problem = seminorm.dummy_problem(loss.smooth_eval, nonsmooth, prox, atom_initial)
        problems.append(problem)
        solvers.append(algorithms.FISTA(problem))
    return problems, solvers, losses

def blockwise(semi, Y, p=None, initial=None):
    problems, solvers, losses = dual_blocks(semi, Y)
    current_resid = Y.copy() 
    for atom, problem in zip(semi.atoms, problems):
        current_resid -= atom.multiply_by_DT(problem.coefs)

    for i in range(5):
        for atom, problem, loss, solver in zip(semi.atoms, problems, losses, solvers):
            old_coefs = problem.coefs.copy()
            adjusted_resid = current_resid + atom.multiply_by_DT(old_coefs)
            loss.set_Y(adjusted_resid)
            h = solver.fit(max_its=800,tol=1e-10)
            current_resid -= atom.multiply_by_DT(problem.coefs)
            primal_soln = current_resid #semi.primal_from_dual(Y, dual_soln)
            print primal_soln, problem.coefs, loss.Y
            print p.obj(primal_soln)
        pylab.clf()
        pylab.figure(num=1)
        pylab.scatter(np.arange(Y.shape[0]), Y, c='r')
        pylab.plot(primal_soln, c='g')
        pylab.draw()

    return primal_soln

def test1():
    import numpy as np
    import pylab
    from scipy import sparse

    from regreg.algorithms import FISTA
    from regreg.atoms import l1norm
    from regreg.seminorm import seminorm, dummy_problem
    from regreg.smooth import signal_approximator

    Y = np.random.standard_normal(500); Y[100:150] += 7; Y[250:300] += 14

    sparsity = l1norm(500, l=1.3)
    #Create D
    D = (np.identity(500) + np.diag([-1]*499,k=1))[:-1]
    D = sparse.csr_matrix(D)
    fused = l1norm(D, l=25.5)

    pen = seminorm(sparsity,fused)
    loss = signal_approximator(Y)
    p = loss.add_seminorm(pen)

    pylab.show()
    soln = blockwise(pen, Y, p)

    #plot solution
    pylab.figure(num=1)
    pylab.scatter(np.arange(Y.shape[0]), Y, c='r')
    pylab.plot(soln, c='g')

def test2():

    import numpy as np
    import pylab
    from scipy import sparse

    from regreg.algorithms import FISTA
    from regreg.atoms import l1norm
    from regreg.seminorm import seminorm, dummy_problem
    from regreg.smooth import signal_approximator

    n1, n2 = l1norm(1), l1norm(1)
    s=seminorm(n1,n2)
    Y = np.array([3.])
    l=signal_approximator(Y)
    blockwise(s, Y,l.add_seminorm(s))

if __name__ == "__main__":
    test2()
