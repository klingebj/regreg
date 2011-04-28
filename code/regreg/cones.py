import numpy as np
from scipy import sparse
from problem import dummy_problem
import atoms

class cone_atom(atoms.seminorm_atom):

    """
    Given a seminorm on :math:`\mathbb{R}^p`, i.e.
    :math:`\beta \mapsto h_K(D\beta+\alpha)`
    this class creates a new seminorm 
    that evaluates on :math:`\mathbb{R}^{p+1}` that
    encodes the cone constraint for the cone

    .. math::

       \left\{(\beta,\epsilon): h_K(D\beta+\alpha) \leq \epsilon\right\}

    """

    tol = 1.0e-06
    def __init__(self, atom, spec, epsilon):
        self.atom = atom
        self.dual_atom = atom.dual_constraint
        seminorm_atom.__init__(self, spec)
        self.epsilon = epsilon
        # an instance with D=I
        # all atoms should (?) be such that this
        # is the conjugate of atom.evaluate

    def evaluate_primal(self, x):
        return self.atom.evaluate(x)

    def evaluate_dual(self, u):
        return self.dual_atom.evaluate_dual

    @property
    def dual_constraint(self):
        # this should be a dual cone
        return primal_dual_pairs[self.atom.__class__](self.m, self.l)

    # the cone constraint is unaffected by scalar multiplication
    def _getl(self):
        return 1. 
    l = property(_getl)

    def evaluate(self, x):
        """
        Return self.atom_I(np.dot(self.atom.D, x)+self.affine_term)

        """
        v = self.atom.evaluate(x[:-1])
        if v <= self.epsilon * (1 + self.tol) and x[-1] == self.epsilon:
            return 0
        return np.inf

    def evaluate_dual(self, u_eps):
        return self.atom.evaluate_dual(u)

    def primal_prox(self, x,  L=1):
        r"""
        Return (unique) minimizer

        .. math::

            v^{\lambda}(x) = \text{argmin}_{v \in \mathbb{R}^p} \frac{L}{2}
            \|x-v\|^2_2 + \lambda h_K(Dv+\alpha)

        where *p*=x.shape[0], :math:`\lambda` = self.l. 

        This is just self.atom.primal_prox(x - self.affine_term, L) + self.affine_term
        """

        return self.atom.primal_prox(x - self.affine_term, L) + self.affine_term


    def dual_prox(self, u, L=1):
        r"""
        Return a minimizer

        .. math::

            v^{\lambda}(u) \in \text{argmin}_{v \in \mathbb{R}^m} \frac{L}{2}
            \|u-v\|^2_2 \ \text{s.t.} \  \|v\|_{\infty} \leq \lambda

        where *m*=u.shape[0], :math:`\lambda` = self.l. 
        This is just truncation: np.clip(u, -self.l/L, self.l/L).
        """
        return self.atom.dual_prox(u, L)

