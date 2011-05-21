import numpy as np
from scipy import sparse
from algorithms import FISTA, ISTA
from problem import dummy_problem

class container(object):
    """
    A container class for storing/combining seminorm_atom classes
    """
    def __init__(self, loss, *atoms):
        self.loss = loss
        self.atoms = []
        self.dual_atoms = []
        self.primal_shape = -1
        self.dual_segments = []
        self.dual_shapes = []
        for atom in atoms:
            if self.primal_shape == -1:
                self.primal_shape = atom.primal_shape
            else:
                if atom.primal_shape != self.primal_shape:
                    raise ValueError("primal dimensions don't agree")
            self.atoms.append(atom)
            
            dual_atom = atom.dual_seminorm
            dual_atom.l = atom.l
            dual_atom.constraint = np.bitwise_not(atom.constraint)
            self.dual_atoms.append(dual_atom)
            
            self.dual_shapes += [atom.dual_shape]
        self.dual_dtype = np.dtype([('dual_%d' % i, np.float, shape) 
                                    for i, shape in enumerate(self.dual_shapes)])
        self.dual_segments = self.dual_dtype.names 

    def __add__(self,y):
        #Combine two seminorms
        def atoms():
            for obj in [self, y]:
                for atom in obj.atoms:
                    yield atom
        return container(*atoms())

    def evaluate_dual_atoms(self, u):
        out = 0.
        # XXX dtype manipulations -- would be nice not to have to do this
        u = u.view(self.dual_dtype).reshape(())

        for atom, segment in zip(self.dual_atoms, self.dual_segments):
            if atom.constraint:
                # atom.evaluate_seminorm already has a factor of atom.l in it
                # This should probably be added as evaluate_primal_constraint in atoms.py
                if atom.evaluate_seminorm(u[segment]) > (1+1e-10)*(atom.l**2):
                    return np.inf
                
            else:
                out += atom.evaluate_seminorm(u[segment])
        return out


    def evaluate_primal_atoms(self, x):
        out = 0.
        for atom in self.atoms:
            if atom.constraint:
                # atom.evaluate_seminorm already has a factor of atom.l in it
                # This should probably be added as evaluate_primal_constraint in atoms.py
                if atom.evaluate_seminorm(x) > (1+1e-10)*(atom.l**2):
                    return np.inf
            else:
                out += atom.evaluate_seminorm(x)
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

        This is used in the ISTA/FISTA solver loop with :math:`u=z-g/L` when finding
        self.primal_prox, i.e., the signal approximator problem.
        """
        # XXX dtype manipulations -- would be nice not to have to do this

        v = np.empty((), self.dual_dtype)
        u = u.view(self.dual_dtype).reshape(())
        for atom, d_atom, segment in zip(self.atoms, self.dual_atoms, self.dual_segments):
            if atom.constraint:
                v[segment] = d_atom.primal_prox(u[segment], L_D)
            else:
                v[segment] = atom.dual_prox(u[segment], L_D)
        return v.reshape((1,)).view(np.float)

    default_solver = FISTA
    def primal_prox(self, y, L_P=1, with_history=False, debug=False, max_its=5000, tol=1e-14):
        """
        The proximal function for the primal problem
        """
        yL = L_P * y
        if not hasattr(self, 'dualopt'):
            self.dualp = self.dual_problem(yL, L_P=L_P)
            #Approximate Lipschitz constant
            self.dualp.L = 1.05*self.power_LD(debug=debug)
            self.dualopt = container.default_solver(self.dualp)
            self.dualopt.debug = debug
        self._dual_prox_center = yL
        history = self.dualopt.fit(max_its=max_its, min_its=5, tol=tol, backtrack=False)
        if with_history:
            return self.primal_from_dual(y, self.dualopt.problem.coefs/L_P), history
        else:
            return self.primal_from_dual(y, self.dualopt.problem.coefs/L_P)

    def power_LD(self,max_its=500,tol=1e-8, debug=False):
        """
        Approximate the Lipschitz constant for the dual problem using power iterations
        """
        v = np.random.standard_normal(self.primal_shape)
        z = np.zeros((), self.dual_dtype)
        old_norm = 0.
        norm = 1.
        itercount = 0
        while np.fabs(norm-old_norm)/norm > tol and itercount < max_its:
            z = np.zeros(z.shape, z.dtype)
            for atom, segment in zip(self.atoms, self.dual_segments):
                z[segment] += atom.linear_map(v)
            v *= 0.
            for atom, segment in zip(self.atoms, self.dual_segments):
                v += atom.adjoint_map(z[segment])
            old_norm = norm
            norm = np.linalg.norm(v)
            v /= norm
            if debug:
                print "L", norm
            itercount += 1
        return norm

    def primal_from_dual(self, y, u):
        """
        Calculate the primal coefficients from the dual coefficients
        """
        x = y * 1.
        u = u.view(self.dual_dtype).reshape(())
        for atom, segment in zip(self.atoms, self.dual_segments):
            x -= atom.adjoint_map(u[segment])
        return x

    def dual_problem(self, y, L_P=1, initial=None):
        """
        Return a problem instance of the dual
        prox problem with a given y value.
        """
        self._dual_prox_center = y
        if initial is None:
            z = np.zeros((), self.dual_dtype)
            for segment in self.dual_segments:
                z[segment] += np.random.standard_normal(z[segment].shape)

            # XXX dtype manipulations -- would be nice not to have to do this
            z = z.reshape((1,)).view(np.float)
            initial = self.dual_prox(z, 1.)
        nonsmooth = self.evaluate_dual_atoms
        prox = self.dual_prox
        return dummy_problem(self._dual_smooth_eval, nonsmooth, prox, initial, 1.)



    def _dual_smooth_eval(self,v,mode='both'):

        """
        The smooth component and/or gradient of the dual objective        
        """
        
        # XXX dtype manipulations -- would be nice not to have to do this
        v = v.view(self.dual_dtype).reshape(())

        # residual is the residual from the fit
        residual = self.primal_from_dual(self._dual_prox_center, v)
        affine_objective = 0
        if mode == 'func':
            for atom, segment in zip(self.atoms, self.dual_segments):
                affine_objective += atom.affine_objective(v[segment])
            return (residual**2).sum() / 2. + affine_objective
        elif mode == 'both' or mode == 'grad':
            g = np.zeros((), self.dual_dtype)
            for atom, segment in zip(self.atoms, self.dual_segments):
                g[segment] = -atom.affine_map(residual)
                affine_objective += atom.affine_objective(v[segment])
            if mode == 'grad':
                # XXX dtype manipulations -- would be nice not to have to do this
                return g.reshape((1,)).view(np.float)
            if mode == 'both':
                # XXX dtype manipulations -- would be nice not to have to do this
                return (residual**2).sum() / 2. + affine_objective, g.reshape((1,)).view(np.float)
        else:
            raise ValueError("Mode not specified correctly")





    def problem(self, smooth_multiplier=1., initial=None):

        prox = self.primal_prox
        nonsmooth = self.evaluate_primal_atoms
        if initial is None:
            initial = np.random.standard_normal(self.primal_shape)
            initial = initial/np.linalg.norm(initial)
        if nonsmooth(initial) + self.loss.smooth_eval(initial,mode='func') == np.inf:
            raise ValueError('initial point is not feasible')

        
        return dummy_problem(self.loss.smooth_eval, nonsmooth, prox, initial, smooth_multiplier)



