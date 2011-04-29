import numpy as np
from scipy import sparse
from algorithms import FISTA, ISTA
from problem import dummy_problem
from atoms import primal_dual_seminorm_pairs

class constraint(object):
    """
    A constraint container class for storing/combining constraint_atom classes
    specified in terms of the conjugate of a smooth convex function
    and some atoms. 

    This class solves the problem

    .. math::

       \text{minimize}_{u_i} {\cal L}^*(-\sum_i D_i^Tu_i) + \sum_i (u_i'\alpha_i + \lambda_i h_i(u_i)

    where :math:`h_i` are the seminorms dual to the atoms.

    The conjugate should be smooth
    """
    def __init__(self, conjugate, *atoms):
        self.conjugate = conjugate
        self.atoms = []
        self.dual_atoms = []
        self.primal_dim = -1
        self.segments = []
        idx = 0
        for atom in atoms:
            if self.primal_dim < 0:
                self.primal_dim = atom.p
            else:
                if atom.p != self.primal_dim:
                    raise ValueError("primal dimensions don't agree")
            self.atoms.append(atom)
            dual_atom = primal_dual_seminorm_pairs[atom.__class__](atom.m, atom.l)
            self.dual_atoms.append(dual_atom)
            self.segments.append(slice(idx, idx+atom.m))
            idx += atom.m
        self.total_dual = idx

    def linear_term(self, u):
        lterm = 0
        for atom, segment in zip(self.atoms, self.segments):
            lterm += atom.multiply_by_DT(u[segment])
        return lterm
    def smooth_eval(self, u, mode='both'):
        linear_term = self.linear_term(u)
        if mode == 'both':
            v, g = self.conjugate.smooth_eval(-linear_term, mode='both')
            grad = np.empty(u.shape)
            for atom, segment in zip(self.atoms, self.segments):
                grad[segment] = -atom.affine_map(g)
                if atom.affine_term is not None:
                    v -= np.dot(atom.affine_term, u[segment])
            return v, grad
        elif mode == 'grad':
            g = self.conjugate.smooth_eval(-linear_term, mode='grad')
            grad = np.empty(u.shape)
            for atom, segment in zip(self.atoms, self.segments):
                grad[segment] = -atom.affine_map(g)
            return grad 
        elif mode == 'func':
            v = self.conjugate.smooth_eval(-linear_term, mode='func')
            for atom, segment in zip(self.atoms, self.segments):
                if atom.affine_term is not None:
                    v -= np.dot(atom.affine_term, u[segment])
            return v
        else:
            raise ValueError("mode incorrectly specified")


    def __add__(self,y):
        #Combine two constraints
        def atoms():
            for obj in [self, y]:
                for atom in obj.atoms:
                    yield atom
        return constraint(*atoms())

    def evaluate_constraint(self, x):
        out = 0.
        for atom in self.atoms:
            out += atom.evaluate_constraint(x)
        return out
    
    def evaluate_dual_seminorm(self, u):
        out = 0.
        for dual_atom, segment in zip(self.dual_atoms, self.segments):
            out += dual_atom.evaluate_seminorm(u[segment])
        return out
    
    def dual_prox(self, u, L_P=1, with_history=False, debug=False, max_its=5000, tol=1e-14):
        """
        Return (unique) minimizer

        .. math::

           v^{\lambda}(u) = \text{argmin}_{v \in \real^m} \frac{L}{2}
           \|v-u\|^2_2 + \sum_i \lambda_i h_i(v)

        where *m*=u.shape[0]=np.sum(self.dual_dims), :math:`M`=self.M
        and :math:`h_i` is the seminorm of 
        self.dual_atoms[i] and 
        :math:`\lambda_i`=self.atoms[i].l.

        """
        value = np.empty(u.shape)
        for dual_atom, segment in zip(self.dual_atoms, self.segments):
            value[segment] = dual_atom.primal_prox(u[segment], L_P)
        return value

    def primal_from_dual(self, u):
        """
        Calculate the primal coefficients from the dual coefficients
        """
        linear_term = self.linear_term(u)
        return self.conjugate.smooth_eval(-linear_term, mode='grad')

    def dual_problem(self, smooth_multiplier=1., initial=None):
        prox = self.dual_prox
        nonsmooth = self.evaluate_dual_seminorm
        smooth_eval = self.smooth_eval
        if initial is None:
            initial = np.random.standard_normal(self.total_dual)
        if nonsmooth(initial) + smooth_eval(initial,mode='func') == np.inf:
            raise ValueError('initial point is not feasible')
        return dummy_problem(smooth_eval, nonsmooth, prox, initial, smooth_multiplier)



