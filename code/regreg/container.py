import numpy as np
from scipy import sparse
from algorithms import FISTA
from composite import composite
from conjugate import conjugate

class container(object):
    """
    A container class for storing/combining seminorm_atom classes
    """
    def __init__(self, loss, *atoms):
        self.loss = loss
        self.atoms = []
        self.primal_shape = -1
        self.dual_segments = []
        self.dual_shapes = []
        self.dual_atoms = []
        for atom in atoms:
            if self.primal_shape == -1:
                self.primal_shape = atom.primal_shape
            else:
                if atom.primal_shape != self.primal_shape:
                    raise ValueError("primal dimensions don't agree")
            self.atoms.append(atom)
            self.dual_atoms.append(atom.dual)

            self.dual_shapes += [atom.dual_shape]
        self.dual_dtype = np.dtype([('dual_%d' % i, np.float, shape) 
                                    for i, shape in enumerate(self.dual_shapes)])
        self.dual_segments = self.dual_dtype.names 

    def __add__(self,y):
        #Combine two seminorms
        raise NotImplementedError
        def atoms():
            for obj in [self, y]:
                for atom in obj.atoms:
                    yield atom
        return container(*atoms())

    def evaluate_dual_atoms(self, u, check_feasibility=False):
        out = 0.
        # XXX dtype manipulations -- would be nice not to have to do this
        u = u.view(self.dual_dtype).reshape(())
        for dual_atom, segment in zip(self.dual_atoms, self.dual_segments):
            transform, atom = dual_atom
            out += atom.nonsmooth_objective(u[segment],
                                            check_feasibility=check_feasibility)
        return out

    def evaluate_primal_atoms(self, x, check_feasibility=False):
        out = 0.
        for atom in self.atoms:
            out += atom.nonsmooth_objective(x,
                                            check_feasibility=check_feasibility)
        return out
    
    def dual_prox(self, u, lipshitz_D=1.):
        """
        Return (unique) minimizer

        .. math::

           v^{\lambda}(u) = \text{argmin}_{v \in \real^m} \frac{1}{2}
           \|v-u\|^2_2  s.t.  h^*_i(v) \leq \infty, 0 \leq i \leq M-1

        where *m*=u.shape[0]=np.sum(self.dual_dims), :math:`M`=self.M
        and :math:`h^*_i` is the conjugate of 
        self.atoms[i].lagrange * self.atoms[i].evaluate and 
        :math:`\lambda_i`=self.atoms[i].lagrange

        This is used in the ISTA/FISTA solver loop with :math:`u=z-g/L` when finding
        self.primal_prox, i.e., the signal approximator problem.
        """
        # XXX dtype manipulations -- would be nice not to have to do this

        v = np.empty((), self.dual_dtype)
        u = u.view(self.dual_dtype).reshape(())
        for dual_atom, segment in zip(self.dual_atoms, self.dual_segments):
            transform, atom = dual_atom
            v[segment] = atom.proximal(u[segment], lipshitz_D)
        return v.reshape((1,)).view(np.float)

    default_solver = FISTA
    def primal_prox(self, y, lipshitz_P=1, prox_control=None):
        #def primal_prox(self, y, lipshitz_P=1, with_history=False, debug=False, max_its=5000, tol=1e-14):
        """
        The proximal function for the primal problem
        """

        #Default fitting parameters
        prox_defaults = {'max_its': 5000,
                         'min_its': 5,
                         'return_objective_hist': False,
                         'tol': 1e-14,
                         'debug':False}
        
        if prox_control is not None:
            prox_defaults.update(prox_control)
        prox_control = prox_defaults

        yL = lipshitz_P * y
        if not hasattr(self, 'dualopt'):
            self.dualp = self.dual_composite(yL, lipshitz_P=lipshitz_P)
            #Approximate Lipshitz constant
            self.dual_reference_lipshitz = 1.05*self.power_LD(debug=prox_control['debug'])
            self.dualopt = container.default_solver(self.dualp)
            self.dualopt.debug = prox_control['debug']

        self.dualopt.composite.smooth_multiplier = 1./lipshitz_P
        self.dualp.lipshitz = self.dual_reference_lipshitz / lipshitz_P

        self._dual_prox_center = yL
        history = self.dualopt.fit(**prox_control)
        if prox_control['return_objective_hist']:
            return self.primal_from_dual(y, self.dualopt.composite.coefs/lipshitz_P), history
        else:
            return self.primal_from_dual(y, self.dualopt.composite.coefs/lipshitz_P)

    def power_LD(self,max_its=500,tol=1e-8, debug=False):
        """
        Approximate the Lipshitz constant for the dual problem using power iterations
        """
        v = np.random.standard_normal(self.primal_shape)
        z = np.zeros((), self.dual_dtype)
        old_norm = 0.
        norm = 1.
        itercount = 0
        while np.fabs(norm-old_norm)/norm > tol and itercount < max_its:
            z = np.zeros(z.shape, z.dtype)
            for dual_atom, segment in zip(self.dual_atoms, self.dual_segments):
                transform, _ = dual_atom
                z[segment] += transform.linear_map(v)
            v *= 0.
            for dual_atom, segment in zip(self.dual_atoms, self.dual_segments):
                transform, _ = dual_atom
                v += transform.adjoint_map(z[segment])
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
        for dual_atom, segment in zip(self.dual_atoms, self.dual_segments):
            transform, _ = dual_atom
            x -= transform.adjoint_map(u[segment])
        return x

    def dual_composite(self, y, lipshitz_P=1, initial=None):
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
            initial = self.dual_prox(z, 1./lipshitz_P)
        nonsmooth_objective = self.evaluate_dual_atoms
        prox = self.dual_prox
        return composite(self._dual_smooth_objective, nonsmooth_objective, prox, initial, 1./lipshitz_P)

    def _dual_smooth_objective(self,v,mode='both', check_feasibility=False):

        """
        The smooth component and/or gradient of the dual objective        
        """
        
        # XXX dtype manipulations -- would be nice not to have to do this
        v = v.view(self.dual_dtype).reshape(())

        # residual is the residual from the fit
        residual = self.primal_from_dual(self._dual_prox_center, v)

        if mode == 'func':
            return (residual**2).sum() / 2.
        elif mode == 'both' or mode == 'grad':
            g = np.zeros((), self.dual_dtype)
            for dual_atom, segment in zip(self.dual_atoms, self.dual_segments):
                transform, _ = dual_atom
                g[segment] = -transform.linear_map(residual)
            if mode == 'grad':
                # XXX dtype manipulations -- would be nice not to have to do this
                return g.reshape((1,)).view(np.float)
            if mode == 'both':
                # XXX dtype manipulations -- would be nice not to have to do this
                return (residual**2).sum() / 2., g.reshape((1,)).view(np.float)
        else:
            raise ValueError("Mode not specified correctly")


    def conjugate_linear_term(self, u):
        lterm = 0
        # XXX dtype manipulations -- would be nice not to have to do this
        u = u.view(self.dual_dtype).reshape(())
        for dual_atom, segment in zip(self.dual_atoms, self.dual_segments):
            transform, _ = dual_atom
            lterm += transform.adjoint_map(u[segment])
        return lterm

    def conjugate_primal_from_dual(self, u):
        """
        Calculate the primal coefficients from the dual coefficients
        """
        # XXX has this changed now that atoms "keep" their
        # own linear terms?
        linear_term = self.conjugate_linear_term(-u)
        return self.conjugate.smooth_objective(linear_term, mode='grad')


    def conjugate_smooth_objective(self, u, mode='both', check_feasibility=False):
        linear_term = self.conjugate_linear_term(u)
        # XXX dtype manipulations -- would be nice not to have to do this
        u = u.view(self.dual_dtype).reshape(())
        if mode == 'both':
            v, g = self.conjugate.smooth_objective(-linear_term, mode='both')
            grad = np.empty((), self.dual_dtype)
            for dual_atom, segment in zip(self.dual_atoms, self.dual_segments):
                transform, _ = dual_atom
                grad[segment] = -transform.linear_map(g)
            # XXX dtype manipulations -- would be nice not to have to do this
            return v, grad.reshape((1,)).view(np.float) 
        elif mode == 'grad':
            g = self.conjugate.smooth_objective(-linear_term, mode='grad')
            grad = np.empty((), self.dual_dtype)
            for dual_atom, segment in zip(self.dual_atoms, self.dual_segments):
                transform, _ = dual_atom
                grad[segment] = -transform.linear_map(g)
            # XXX dtype manipulations -- would be nice not to have to do this
            return grad.reshape((1,)).view(np.float) 
        elif mode == 'func':
            v = self.conjugate.smooth_objective(-linear_term, mode='func')
            return v
        else:
            raise ValueError("mode incorrectly specified")


    def conjugate_composite(self, conj=None, initial=None, smooth_multiplier=1., conjugate_tol=1.0e-08, epsilon=0.):
        """
        Create a composite object for solving the conjugate problem
        """

        if conj is not None:
            self.conjugate = conj
        if not hasattr(self, 'conjugate'):
            #If the conjugate of the loss function is not provided use the generic solver
            self.conjugate = conjugate(self.loss, tol=conjugate_tol,
                                       epsilon=epsilon)

        prox = self.dual_prox
        nonsmooth_objective = self.evaluate_dual_atoms

        if initial is None:
            z = np.zeros((), self.dual_dtype)
            for segment in self.dual_segments:
                z[segment] += np.random.standard_normal(z[segment].shape)

            # XXX dtype manipulations -- would be nice not to have to do this
            z = z.reshape((1,)).view(np.float)
            initial = self.dual_prox(z, 1.)

        return composite(self.conjugate_smooth_objective, nonsmooth_objective, prox, initial, smooth_multiplier)
        

    def composite(self, smooth_multiplier=1., initial=None):
        """
        Create a composite object for solving the general problem with the two-loop algorithm
        """

        if initial is None:
            initial = np.random.standard_normal(self.atoms[0].primal_shape)
        prox = self.primal_prox
        nonsmooth_objective = self.evaluate_primal_atoms
        
        return composite(self.loss.smooth_objective, nonsmooth_objective, prox, initial, smooth_multiplier)

