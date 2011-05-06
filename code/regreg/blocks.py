import numpy as np
import seminorm, smooth, algorithms
import pylab

#XXX TODO use the constraint module here

class Block(object):

    r"""
    A Block can solve a signal approximator problem of the form

    .. math::

       \text{minimize} \frac{1}{2} \|Y - D^Tu\|^2_2 \ \text{s.t.} \ h^*(u)
       \leq \lambda

    """

    def __init__(self, atom, initial=None):
        self.atom = atom
        Y = np.zeros(atom.p)
        if initial is None:
            initial = np.zeros(atom.m) 
        if not atom.noneD:
            self.loss = smooth.smooth_function(smooth.squaredloss(atom.D.T, Y))
        else:
            self.loss = smooth.smooth_function(smooth.signal_approximator(Y))

        dual_atom = atom.dual_constraint
        prox = dual_atom.primal_prox
        nonsmooth = dual_atom.evaluate_constraint
        if nonsmooth(initial) == np.inf:
            raise ValueError('initial point is not feasible')
        
        self.problem = seminorm.dummy_problem(self.loss.smooth_eval, nonsmooth, prox, initial)

    def fit(self, *solver_args, **solver_kw):
        if not hasattr(self, '_solver'):
            self._solver = algorithms.FISTA(self.problem)
        self._solver.fit(*solver_args, **solver_kw)
        return self.problem.coefs

    def set_coefs(self, coefs):
        self.problem.coefs[:] = coefs

    def get_coefs(self):
        return self.problem.coefs
    coefs = property(get_coefs, set_coefs)
    

    def set_Y(self, Y):
        self.loss.atoms[0].Y[:] = Y
    def get_Y(self):
        return self.loss.atoms[0].Y
    Y = property(get_Y, set_Y)
    
def dual_blocks(semi, Y, initial=None):
    blocks = []
    if initial is None:
        initial = [None] * len(semi.atoms)
    for atom, atom_initial in zip(semi.atoms, initial):
        blocks.append(Block(atom, initial=atom_initial))
    return blocks

def blockwise(semi, Y, initial=None, max_its=50, tol=1.0e-06,
              min_its=5):

    blocks = dual_blocks(semi, Y)
    current_resid = Y.copy() 
    adjusted_resid = Y.copy()
    primal_soln = current_resid
    for block in blocks:
        current_resid -= block.atom.adjoint_map(block.coefs)

    for itercount in range(max_its):
        for block in blocks:

            # XXX for a distributed version
            # each block can maintain a copy of
            # current resid that should be reset by a scatter
            # before each block updates its coefficients

            block.Y = current_resid + block.atom.adjoint_map(block.coefs)
            block.fit(max_its=800,tol=1e-10)
            current_resid[:] = block.Y - block.atom.adjoint_map(block.coefs)
            if np.linalg.norm(primal_soln - current_resid) / np.max([1.,np.linalg.norm(current_resid)]) < tol and itercount >= min_its:
                return current_resid
            primal_soln = current_resid
    return primal_soln

def test1():
    import numpy as np
    import pylab
    from scipy import sparse

    from regreg.algorithms import FISTA
    from regreg.atoms import l1norm
    from regreg.seminorm import seminorm, dummy_problem
    from regreg.smooth import signal_approximator, smooth_function

    Y = np.random.standard_normal(500); Y[100:150] += 7; Y[250:300] += 14

    sparsity = l1norm(500, l=1.0)
    #Create D
    D = (np.identity(500) + np.diag([-1]*499,k=1))[:-1]
    D = sparse.csr_matrix(D)
    fused = l1norm(D, l=19.5)

    pen = seminorm(sparsity,fused)
    loss = smooth_function(signal_approximator(Y))
    p = loss.add_seminorm(pen)

    
    soln1 = blockwise(pen, Y)

    solver = FISTA(p)
    solver.fit(max_its=800,tol=1e-10)
    soln2 = solver.problem.coefs

    #plot solution
    pylab.figure(num=1)
    pylab.clf()
    pylab.scatter(np.arange(Y.shape[0]), Y, c='r')
    pylab.plot(soln1, c='y', linewidth=6)
    pylab.plot(soln2, c='b', linewidth=2)


def test2():

    import numpy as np
    import pylab
    from scipy import sparse

    from regreg.algorithms import FISTA
    from regreg.atoms import l1norm
    from regreg.seminorm import seminorm, dummy_problem
    from regreg.smooth import signal_approximator, smooth_function

    n1, n2 = l1norm(1), l1norm(1)
    s=seminorm(n1,n2)
    Y = np.array([30.])
    l=smooth_function(signal_approximator(Y))
    blockwise(s, Y,l.add_seminorm(s))


if __name__ == "__main__":
    test1()
