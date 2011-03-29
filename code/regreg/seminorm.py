import numpy as np
from scipy import sparse
from algorithms import FISTA, ISTA

class seminorm(object):
    """
    A seminorm container class for storing/combining seminorm_atom classes
    """
    def __init__(self, *atoms):
        self.atoms = []
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
            self.segments.append(slice(idx, idx+atom.m))
            idx += atom.m
        self.total_dual = idx
        self._dual_variables = np.empty(self.total_dual)
        self._dual_grad = np.empty(self.total_dual)
        self._pseudo_response = np.empty(self.total_dual)
        self._pseudo_response_segments = []
        self._dual_segments = []
        self._dual_grad_segments = []
        for i, segment in self.segments:
            self._dual_segments[i] = self._dual_variables[segment]
            self._dual_grad_segments[i] = self._dual_grad[segment]
            self._pseudo_response_segments[i] = self._pseudo_response[segment]
        self._primal_variables = np.empty(self.primal_dim)
        
    def __add__(self,y):
        #Combine two seminorms
        def atoms():
            for obj in [self, y]:
                for atom in obj.atoms:
                    yield atom
        return seminorm(*atoms())

    def evaluate(self, x):
        out = 0.
        for atom in self.atoms:
            out += atom.evaluate(x)
        return out
    
    def evaluate_dual(self, u):
        out = 0.
        if id(u) == id(self._dual_variables):
            for atom, segment in zip(self.atoms, self._dual_segments):
                out += atom.evaluate_dual(segment)
        else:
            for atom, segment in zip(self.atoms, self.segments):
                out += atom.evaluate_dual(u[segment])
        return out
    
    def dual_prox(self, u, L_D=1.):
        """
        Return (unique) minimizer

        .. math::

           v^{\lambda}(u) = \text{argmin}_{v \in \real^m} \frac{1}{2}
           \|v-u\|^2_2  s.t.  h^*_i(v) \leq \infty, 0 \leq i \leq M-1

        where *m*=u.shape[0]=np.sum(self.dual_dims), :math:`M`=self.M
        and :math:`h^*_i` is the conjugate of 
        self.atoms[i].l * self.atoms[i].evaluate and 
        :math:`\lambda_i`=self.atoms[i].l.

        This is used in the inner loop with :math:`u=z-g/L` when finding
        self.primal_prox, i.e., the signal approximator problem.
        """
        if id(u) != id(self._pseudo_response):
            v = np.empty(u.shape)
            for atom, segment in zip(self.atoms, self.segments):
                v[segment] = atom.dual_prox(u[segment], L_D)
        else:
            for atom, pseudo_response, dual in zip(self.atoms,
                                                   self._pseudo_response_segments,
                                                   self._dual_segments):
                dual[:] = atom.dual_prox(pseudo_response, L_D)
        return v

    default_solver = FISTA
    def primal_prox(self, y, L_P=1, with_history=False, debug=False,tol=1e-15):
        """
        The proximal function for the primal problem
        """
        yL = L_P * y
        if not hasattr(self, 'dualopt'):
            self.dualp = self.dual_problem(yL, L_P=L_P)
            #Approximate Lipschitz constant
            self.dualp.L = 1.1*self.power_LD(debug=debug)
            self.dualopt = seminorm.default_solver(self.dualp)
            self.dualopt.debug = debug
        self._dual_prox_center = yL
        history = self.dualopt.fit(max_its=5000, min_its=5, tol=tol, backtrack=False)
        self._dual_variables[:] = self.dualopt.problem.coefs
        if with_history:
            return self.primal_from_dual(y, self.dualopt.problem.coefs/L_P), history
        else:
            return self.primal_from_dual(y, self.dualopt.problem.coefs/L_P)

    def power_LD(self,max_its=50,tol=1e-5, debug=False):
        """
        Approximate the Lipschitz constant for the dual problem using power iterations
        """
        v = np.random.standard_normal(self.primal_dim)
        z = np.zeros(self.total_dual)
        old_norm = 0.
        norm = 1.
        itercount = 0
        while np.fabs(norm-old_norm)/norm > tol and itercount < max_its:
            z *= 0.
            for atom, segment in zip(self.atoms, self.segments):
                z[segment] += atom.multiply_by_D(v)
            v *= 0.
            for atom, segment in zip(self.atoms, self.segments):
                v += atom.multiply_by_DT(z[segment])
            old_norm = norm
            norm = np.linalg.norm(v)
            v /= norm
            if debug:
                print "L", norm
            itercount += 1
        return norm
        #return np.sqrt(norm)

    def primal_from_dual(self, y, u):
        """
        Calculate the primal coefficients from the dual coefficients
        """
        if id(u) != id(self._dual_variables):
            x = y * 1.
            for atom, segment in zip(self.atoms, self.segments):
                x -= atom.multiply_by_DT(u[segment])
            return x
        else:
            x = self._primal_variables
            x[:] = y
            for atom, dual in zip(self.atoms, self._dual_segments):
                x -= atom.multiply_by_DT(dual)
            return x
            

    def dual_problem(self, y, L_P=1, initial=None):
        """
        Return a problem instance of the dual
        prox problem with a given y value.
        """
        self._dual_prox_center = y
        if initial is None:
            z = np.random.standard_normal(self.total_dual)
            initial = self.dual_prox(z, 1.)
        nonsmooth = self.evaluate_dual
        prox = self.dual_prox
        return dummy_problem(self._dual_smooth, self._dual_grad_smooth, nonsmooth, prox, initial, 1.)

    def _dual_smooth(self, v):
        """
        The smooth component of the dual objective
        """
        primal = self.primal_from_dual(self._dual_prox_center, v)
        return (primal**2).sum() / 2.

    def _dual_grad_smooth(self, v):
        """
        The gradient of the smooth component of the dual objective
        """
        if id(v) != id(self._dual_variables):
            primal = self.primal_from_dual(self._dual_prox_center, v)
            g = np.zeros(self.total_dual)
            for atom, segment in zip(self.atoms, self.segments):
                g[segment] = -atom.multiply_by_D(primal)
            return g
        else:
            primal = self.primal_from_dual(self._dual_prox_center, self._dual_variables)
            for atom, dual_grad in zip(self.atoms, self._dual_grad_segments):
                dual_grad[:] = -atom.multiply_by_D(primal)
            return dual_grad

    def problem(self, smooth, grad_smooth, smooth_multiplier=1., initial=None):
        prox = self.primal_prox
        nonsmooth = self.evaluate
        if initial is None:
            initial = self.dual_prox(np.random.standard_normal(self.primal_dim))
        return dummy_problem(smooth, grad_smooth, nonsmooth, prox, initial, smooth_multiplier)



class dummy_problem(object):
    """
    A generic way to specify a problem
    """
    def __init__(self, smooth, grad_smooth, nonsmooth, prox, initial, smooth_multiplier=1):
        self.initial = initial.copy()
        self._pseudo_response = np.empty(self.initial.shape)
        self.coefs = initial.copy()
        self.obj_smooth = smooth
        self.obj_rough = nonsmooth
        self._grad = grad_smooth
        self._prox = prox
        self.smooth_multiplier = smooth_multiplier

    def obj(self, x):
        return self.smooth_multiplier * self.obj_smooth(x) + self.obj_rough(x)

    def grad(self, x):
        return self.smooth_multiplier * self._grad(x)

    def proximal(self, x, g, L, tol=None):
        self._pseudo_response[:] = x - g / L
        # XXX maybe allow atoms to have tol -- which are automatically
        # satisfied because they solve the problems to machine precision
        if tol is None:
            return self._prox(self._pseudo_response, L)
        else:
            return self._prox(self._pseudo_response, L, tol=tol)
