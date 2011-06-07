import numpy as np
from scipy import sparse
from algorithms import FISTA
from problem import dummy_problem
from conjugate import conjugate
from smooth import smooth_function



#TODO: this is only written for linear compositions, need to add affine

class admm_problem(object):

    """
    A class for solving the generic problem

    .. math::
    
       \mbox{argmin}_\beta \quad \mathcal{L}(\beta) + \sum_i \lambda_i h_{K_i}(D_i \beta)

    using the ADMM algorithm with the substitution $z_i = D_i \beta$ and solving the augmented lagrangian

    .. math::

       \mbox{argmin}_\beta \mathcal{L}(\beta) + \sum_i \lambda_i h_{K_i}(z_i) + \sum_i u_i^T(z_i - D_i \beta) + \frac{\rho}{2} \sum_i \|z_i - D_i\beta\|_2^2  
    """
        
    def __init__(self, container):

        self.smooth = container.loss
        self.atoms = container.atoms

        self.beta = np.zeros(self.atoms[0].primal_shape)
        self.us = [np.zeros(atom.dual_shape) for atom in self.atoms]
        self.node_problems = [node_problem(atom, self.beta, u) for atom, u in zip(self.atoms, self.us)]
        self.rho = 1.

        self.prob = self.problem()
        self.solver = FISTA(self.prob)

    def fit(self, tol = 1e-6, max_its = 500, debug=False):
        coef_change = 1.
        itercount = 0
        while coef_change > tol and itercount <= max_its:
            old_beta = self.beta.copy()
            self.solve_beta()
            self.solve_z()
            self.solve_u()
            coef_change = np.linalg.norm(self.beta - old_beta) / np.linalg.norm(self.beta)
            itercount += 1
            if debug:
                print itercount, coef_change
        

    def solve_beta(self, tol=1e-6):
        self.solver.fit(tol=tol)
        self.beta = self.solver.problem.coefs

    def solve_z(self):
        for problem, u in zip(self.node_problems, self.us):
            problem.u = u
            problem.beta = self.beta
            problem.fit()

    def solve_u(self):
        for u, problem, atom in zip(self.us, self.node_problems, self.atoms):
            u += (problem.coefs - atom.linear_map(self.beta))

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
                f += (self.rho/2.) * np.linalg.norm(problem.coefs - affine + u)**2 
                g += - self.rho * atom.adjoint_map(problem.coefs - affine + u) 
            return f, g
        elif mode == 'func':
            f = 0
            for u, problem, atom in zip(self.us, self.node_problems, self.atoms):
                affine = atom.linear_map(x)
                f += (self.rho/2.) * np.linalg.norm(problem.coefs - affine + u)**2 
            return f
        elif mode == 'grad':
            g = 0
            for u, problem, atom in zip(self.us, self.node_problems, self.atoms):
                affine = atom.linear_map(x)
                g += - self.rho * atom.adjoint_map(problem.coefs - affine + u) 
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


    def fit(self, debug=False):
        self.solver.debug = debug
        self.solver.fit()
        self.coefs = self.solver.problem.coefs

    def smooth_eval(self, x, mode='both'):
        if mode == 'both':
            return (self.rho/2.) * np.linalg.norm(x - self.affine + self.u)**2, (self.rho)*(x - self.affine + self.u)
        elif mode == 'func':
            return  (self.rho/2.) * np.linalg.norm(x - self.affine + self.u)**2
        elif mode == 'grad':
            return  (self.rho)*(x - self.affine + self.u)
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
            if hasattr(self.atom, 'atom'):
                return dummy_problem(self.smooth_eval, self.atom.atom.evaluate_seminorm, self.atom.atom.primal_prox, initial, smooth_multiplier)
            else:
                return dummy_problem(self.smooth_eval, self.atom.evaluate_seminorm, self.atom.primal_prox, initial, smooth_multiplier) 
            
#This is a lazy, temporary fix, to embed a smooth_eval into a smooth_function object with primal_shape, etc. We should probably have the zero function seminorm atom.
class hold_smooth(object):

    def __init__(self, smooth, primal_shape, lagrange=1.):

        self.primal_shape = primal_shape
        self.smooth_eval = smooth
        self.lagrange = 1.
