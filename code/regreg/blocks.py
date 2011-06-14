import numpy as np
import container, smooth, algorithms
from composite import composite
import pylab

class Block(object):

    r"""
    A Block solves a signal approximator problem for a given atom, the form
    of which depends on whether the atom is in constraint mode
    or Lagrange mode. An atom has an affine transform, i.e. a map

    .. math::

       \beta \mapsto D\beta + \alpha

    and a support function. The canonical example, the :math:`\ell_1` norm,
    is the support function of the :math:`\ell_{\infty}` ball.

    If it's in Lagrange mode, the Block solves the following
    dual problem.

    and a support function. The canonical example, the :math:`\ell_1` norm,
    is the support function of the :math:`\ell_{\infty}` ball.

    If it's in Lagrange mode, the Block solves the following
    dual problem.

    .. math::

       \text{minimize} \frac{1}{2} \|Y - D^Tu\|^2_2 - \alpha^T u \ \text{s.t.} \ h^*(u)
       \leq \lambda

    where :math:`h^*(u) \leq \lambda` is the dual constraint of the atom.
    If the atom was the *l1norm*, then the constraint is an
    :math:`\ell_{\infty}` or box constraint.

    If the atom is in constraint mode, the Block solves the following
    dual problem

    .. math::

       \text{minimize} \frac{1}{2} \|Y - D^Tu\|^2_2 - \alpha^T u +
       \lambda h^*(u)

    If the atom was the *l1norm*, then the norm above is
    :math:`\ell_{\infty}` norm.

    """

    def __init__(self, dual, initial=None, response=None):

        self.linear_transform, self.dual_atom = dual
        if response is None:
            response = np.zeros(self.linear_transform.dual_shape)
        if initial is None:
            initial = np.zeros(self.linear_transform.dual_shape)
        else:
            initial = initial.copy()
        if self.linear_transform.linear_operator is not None:
            self.loss = smooth.l2normsq.affine(self.linear_transform.linear_operator.T, -response, coef=0.5, diag=self.linear_transform.diagD)
        else:
            self.loss = smooth.l2normsq.shift(-response, coef=0.5)

        prox = self.dual_atom.proximal
        nonsmooth = self.dual_atom.nonsmooth_objective
        if nonsmooth(initial) == np.inf:
            raise ValueError('initial point is not feasible')
        
        self.composite = composite(self.loss.smooth_objective, nonsmooth, prox, initial)

    def fit(self, *solver_args, **solver_kw):
        if not hasattr(self, '_solver'):
            self._solver = algorithms.FISTA(self.composite)
        self._solver.fit(*solver_args, **solver_kw)
        return self.composite.coefs

    def set_coefs(self, coefs):
        self.composite.coefs[:] = coefs

    def get_coefs(self):
        return self.composite.coefs
    coefs = property(get_coefs, set_coefs)
    
    # response is the negative offset in the loss which is (l2normsq / 2.)
    def set_response(self, response):
        self.loss.affine_transform.affine_offset[:] = -response
    def get_response(self):
        return -self.loss.affine_transform.affine_offset.copy()
    response = property(get_response, set_response)
    
def dual_blocks(atoms, current_resid, initial=None):
    """
    Optional initial dual variables.
    """
    blocks = []
    if initial is None:
        initial = [None] * len(atoms)
    for atom, atom_initial in zip(atoms, initial):
        blocks.append(Block(atom.dual, initial=atom_initial, response=current_resid))
    return blocks

def blockwise(atoms, response, initial=None, max_its=50, tol=1.0e-06,
              min_its=5):

    current_resid = response.copy() 
    blocks = dual_blocks(atoms, current_resid, initial=initial)
    primal_soln = current_resid.copy()
    for block in blocks:
        current_resid -= block.linear_transform.adjoint_map(block.coefs)

    for itercount in range(max_its):
        for block in blocks:

            # XXX for a distributed version
            # each block can maintain a copy of
            # current resid that should be reset by a scatter
            # before each block updates its coefficients

            block.response = current_resid + block.linear_transform.adjoint_map(block.coefs)
            block.fit(max_its=800,tol=1e-10)
            current_resid[:] = block.response - block.linear_transform.adjoint_map(block.coefs)
            if np.linalg.norm(primal_soln - current_resid) / np.max([1.,np.linalg.norm(current_resid)]) < tol and itercount >= min_its:
                return current_resid
            primal_soln = current_resid.copy()
    return primal_soln

def test1():
    import numpy as np
    import pylab
    from scipy import sparse

    from regreg.algorithms import FISTA
    from regreg.atoms import l1norm
    from regreg.container import container
    from regreg.smooth import l2normsq

    Y = np.random.standard_normal(500); Y[100:150] += 7; Y[250:300] += 14

    sparsity = l1norm(500, lagrange=1.0)
    #Create D
    D = (np.identity(500) + np.diag([-1]*499,k=1))[:-1]
    D = sparse.csr_matrix(D)

    fused = l1norm.linear(D, lagrange=19.5)
    loss = l2normsq.shift(-Y, lagrange=0.5)

    p = container(loss, sparsity, fused)
    
    soln1 = blockwise([sparsity, fused], Y)

    solver = FISTA(p)
    solver.fit(max_its=800,tol=1e-10)
    soln2 = solver.composite.coefs

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
    from regreg.container import container
    from regreg.smooth import l2normsq

    n1, n2 = l1norm(1), l1norm(1)
    Y = np.array([30.])
    loss = l2normsq.shift(-Y, lagrange=0.5)
    blockwise([n1, n2], Y)


if __name__ == "__main__":
    test1()
