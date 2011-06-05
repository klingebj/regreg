import numpy as np
from scipy import sparse
from algorithms import FISTA
from problem import dummy_problem
from conjugate import conjugate
from smooth import smooth_function




#TODO: this is only written for linear compositions, need to add affine

class master_problem(object):

    """
    A class for solving the generic problem

    .. math::
    
       \mbox{argmin}_\beta \quad \mathcal{L}(\beta) + \sum_i \lambda_i h_{K_i}(D_i \beta)

    using the ADMM algorithm with the substitution $z_i = D_i \beta$ and solving the augmented lagrangian

    .. math::

       \mbox{argmin}_\beta \mathcal{L}(\beta) + \sum_i \lambda_i h_{K_i}(z_i) + \sum_i u_i^T(z_i - D_i \beta) + \frac{\rho}{2} \sum_i \|z_i - D_i\beta\|_2^2  
    """
        
    def __init__(self, smooth,  *atoms):

        self.smooth = smooth
        self.atoms = atoms
        self._rho = 1.
        self.beta = np.zeros(self.atoms[0].primal_shape[0])
        self.us = [np.zeros(atom.dual_shape) for atom in self.atoms]
        self.node_problems = [node_problem(atom, beta, u) for atom, u in zip(self.atoms, self.us)]

        self.prob = self.problem()
        self.solver = FISTA(self.prob)

    def fit(self):
        for i in range(1000):
            self.solve_beta()
            self.solve_u()
            self.solve_z()
        

    def solve_beta(self):
        self.solver.fit()
        self.beta = self.solver.problem.coefs

    def solve_z(self):
        for problem, u in zip(self.node_problems, self.us):
            problem.u = u
            problem.beta = beta
            problem.fit()

    def solve_u(self):
        for u, problem, atom in zip(self.us, self.node_problems, self.atoms):
            u += self.rho * (problem.coefs - atom.linear_map(self.beta))

    def _get_rho(self):
        return self._rho
    def _set_rho(self, rho):
        self._rho = rho
        for problem in self.node_problems:
            problem.rho = rho
    rho = property(_get_rho, _set_rho)

    def smooth_eval(self, x, mode='both'):

        if mode == 'both':
            f = 0
            g = 0
            for u, problem, atom in zip(self.us, self.node_problems, self.atoms):
                affine = atom.linear_map(x)
                f += (self.rho/2.) * np.linalg.norm(problem.coefs - affine)**2 - np.dot(u, affine)
                g += atom.adjoint_map(self.rho * (problem.coefs - affine) - u)
            return f, g
        elif mode == 'func':
            f = 0
            for u, problem, atom in zip(self.us, self.node_problems, self.atoms):
                affine = atom.linear_map(x)
                f += (self.rho/2.) * np.linalg.norm(problem.coefs - affine)**2 - np.dot(u, affine)
            return f
        elif mode == 'grad':
            g = 0
            for u, problem, atom in zip(self.us, self.node_problems, self.atoms):
                affine = atom.linear_map(x)
                g += atom.adjoint_map(self.rho * (problem.coefs - affine) - u)
            return  g
        else:
            raise ValueError("Mode not specified correctly")

    def problem(self, smooth_multiplier=1., initial=None):
        """
        Create a problem object
        """

        if initial is None:
            initial = self.beta.copy()

        def identity(x):
            return x

        def zero(x):
            return 0.

        return smooth_function(self.smooth, hold_smooth(self.smooth_eval,self.smooth.primal_shape))

           
class node_problem(object):
    """
    A class for storing and updating the node coefficients $z_i = D_i \beta_i$ for a single node
    """

    def __init__(self, atom, beta, u,  rho=1., initial = None):

        self.atom = atom
        self.dual_atom = atom.conjugate
        self.beta = beta
        self._u = u
        self.rho = rho
        if initial is None:
            self.coefs = np.zeros(self.atom.dual_shape)
        else:
            self.coefs = initial

        self.prob = self.problem()
        self.solver = FISTA(self.prob)
        

    def _get_beta(self):
        return self._beta
    def _set_beta(self, beta):
        self._beta = beta
        self.affine = self.atom.linear_map(self._beta)
    beta = property(_get_beta, _set_beta)

    def _get_u(self):
        return self._u
    def _set_u(self, u):
        self._u = u
    u = property(_get_u, _set_u)


    def fit(self, debug=True):
        self.solver.debug = debug
        self.solver.fit()
        self.coefs = self.solver.problem.coefs

    def smooth_eval(self, x, mode='both'):
        if mode == 'both':
            return np.dot(self.u, x) + (self.rho/2.)*np.linalg.norm(x - self.affine)**2, self.u + (self.rho)*(x - self.affine)
        elif mode == 'func':
            return np.dot(self.u, x) + (self.rho/2.)* np.linalg.norm(x - self.affine)**2
        elif mode == 'grad':
            return self.u + (self.rho)*(x - self.affine)
        else:
            raise ValueError("Mode specified incorrectly")



    
    def problem(self, smooth_multiplier=1., initial=None):
        """
        Create a problem object
        """

        if initial is None:
            initial = self.coefs

        if self.atom.constraint:
            return dummy_problem(self.smooth_eval, self.dual_atom.evaluate_dual_constraint, self.dual_atom.dual_prox, initial, smooth_multiplier)
        else:
            return dummy_problem(self.smooth_eval, self.atom.evaluate_seminorm, self.atom.primal_prox, initial, smooth_multiplier) 


#This is a lazy, temporary fix, to embed a smooth_eval into a smooth_function object with primal_shape, etc. We should probably have the zero function seminorm atom.
class hold_smooth(object):

    def __init__(self, smooth, primal_shape, lagrange=1.):

        self.primal_shape = primal_shape
        self.smooth_eval = smooth
        self.lagrange = 1.
