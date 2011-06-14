import numpy as np
from scipy import sparse
from algorithms import FISTA
from composite import composite as composite_class
from conjugate import conjugate
from smooth import smooth_function



#TODO: this is only written for linear compositions, need to add affine

class admm_problem(composite_class):

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

        self.p = len(self.beta)
        self.total_n = np.sum([len(u) for u in self.us])

        self.solver = FISTA(self.composite())

    def fit(self, tol = 1e-6, max_its = 500, debug=False):
        coef_change = 1.
        itercount = 0
        mu = 10.
        while coef_change > tol and itercount <= max_its:
            old_beta = self.beta.copy()
            self.solve_beta()
            self.solve_z()
            self.solve_u()
            #coef_change = np.linalg.norm(self.beta - old_beta) / np.linalg.norm(self.beta)
            coef_change = (self.residual_norm/self.p) + (self.dual_residual_norm/self.total_n)
            if self.residual_norm > mu * self.dual_residual_norm:
                self.rho *= 2.
                for u in self.us:
                    u /= 2.
            elif self.dual_residual_norm > mu * self.residual_norm:
                self.rho /= 2.
                for u in self.us:
                    u *= 2.
            itercount += 1
            if debug:
                print itercount, coef_change, self.rho

        

    def solve_beta(self, tol=1e-6):
        self.solver.fit(tol=tol)
        self.beta = self.solver.composite.coefs

    def solve_z(self):
        self.dual_residual_norm = 0.
        for problem, u in zip(self.node_problems, self.us):
            problem.u = u
            problem.beta = self.beta
            problem.fit()
            self.dual_residual_norm += problem.dual_residual_norm

    def solve_u(self):
        self.residual_norm = 0.
        for u, problem in zip(self.us, self.node_problems):
            u += (problem.coefs - problem.linear_transform.linear_map(self.beta))
            self.residual_norm += np.linalg.norm(problem.coefs - problem.linear_transform.linear_map(self.beta))

            
    def _get_rho(self):
        return self._rho
    def _set_rho(self, rho):
        self._rho = rho
        for problem in self.node_problems:
            problem.rho = rho
    rho = property(_get_rho, _set_rho)

    def smooth_objective(self, x, mode='both',check_feasibility=False):

        if mode == 'both':
            f = 0
            g = 0
            for u, problem in zip(self.us, self.node_problems):
                affine = problem.linear_transform.linear_map(x)
                f += (self.rho/2.) * np.linalg.norm(problem.coefs - affine + u)**2 
                g += - self.rho * problem.linear_transform.adjoint_map(problem.coefs - affine + u) 
            return f, g
        elif mode == 'func':
            f = 0
            for u, problem in zip(self.us, self.node_problems):
                affine = problem.linear_transform.linear_map(x)
                f += (self.rho/2.) * np.linalg.norm(problem.coefs - affine + u)**2 
            return f
        elif mode == 'grad':
            g = 0
            for u, problem in zip(self.us, self.node_problems):
                affine = problem.linear_transform.linear_map(x)
                g += - self.rho * problem.linear_transform.adjoint_map(problem.coefs - affine + u) 
            return  g
        else:
            raise ValueError("Mode not specified correctly")

    def composite(self, smooth_multiplier=1., initial=None):
        """
        Create a problem object
        """

        if initial is None:
            initial = self.beta.copy()

        def identity(x):
            return x

        def zero(x):
            return 0.

        return smooth_function(self.smooth, hold_smooth(self.smooth_objective,self.smooth.primal_shape))

           
class node_problem(composite_class):
    """
    A class for storing and updating the node coefficients $z_i = D_i \beta_i$ for a single node
    """

    def __init__(self, atom, beta, u,  rho=1., initial = None):

        self.atom = atom
        self.linear_transform, self.dual_atom = atom.dual
        self.beta = beta
        self._u = u
        self.rho = rho
        if initial is None:
            self.coefs = np.zeros(self.atom.dual_shape)
        else:
            self.coefs = initial

        self.solver = FISTA(self.composite())
        

    def _get_beta(self):
        return self._beta
    def _set_beta(self, beta):
        self._beta = beta
        self.affine = self.linear_transform.linear_map(self._beta)
    beta = property(_get_beta, _set_beta)

    def _get_u(self):
        return self._u
    def _set_u(self, u):
        self._u = u
    u = property(_get_u, _set_u)


    def fit(self, debug=False):
        self.solver.debug = debug
        self.solver.fit()
        self.dual_residual = self.rho * self.linear_transform.adjoint_map(self.solver.composite.coefs - self.coefs)
        self.dual_residual_norm = np.linalg.norm(self.dual_residual)
        self.coefs = self.solver.composite.coefs

    def smooth_objective(self, x, mode='both',check_feasibility=False):
        if mode == 'both':
            return (self.rho/2.) * np.linalg.norm(x - self.affine + self.u)**2, (self.rho)*(x - self.affine + self.u)
        elif mode == 'func':
            return  (self.rho/2.) * np.linalg.norm(x - self.affine + self.u)**2
        elif mode == 'grad':
            return  (self.rho)*(x - self.affine + self.u)
        else:
            raise ValueError("Mode specified incorrectly")



    
    def composite(self, smooth_multiplier=1., initial=None):
        """
        Create a problem object
        """

        if initial is None:
            initial = self.coefs


        if self.atom.bound is not None:
            return composite_class(self.smooth_objective, self.atom.nonsmooth_objective, self.atom.bound_prox, initial, smooth_multiplier)
            #XXX this needs to be fixed when ADMM is rewritten
            # as currently implemented, the linear_atom cannot set bound/lagrange parameters
            #if hasattr(self.atom, 'bound') and self.atom.bound is not None:
            #return composite_class(self.smooth_objective, self.dual_atom.nonsmooth_objective, self.dual_atom.lagrange_prox, initial, smooth_multiplier)
        else:
            if hasattr(self.atom, 'atom'):
                return composite_class(self.smooth_objective, self.atom.atom.nonsmooth_objective, self.atom.atom.lagrange_prox, initial, smooth_multiplier)
            else:
                return composite_class(self.smooth_objective, self.atom.nonsmooth_objective, self.atom.lagrange_prox, initial, smooth_multiplier) 
            
#This is a lazy, temporary fix, to embed a smooth_objective into a smooth_function object with primal_shape, etc. We should probably have the zero function seminorm atom.
class hold_smooth(object):

    def __init__(self, smooth, primal_shape, lagrange=1.):

        self.primal_shape = primal_shape
        self.smooth_objective = smooth
        self.lagrange = 1.
