import numpy as np
from scipy import sparse
from algorithms import FISTA
from composite import composite
from affine import stack as afstack, identity as afidentity, power_L
from separable import separable
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

    @property
    def dual(self):
        transforms = []
        dual_atoms = []
        for atom in self.atoms:
            t, a = atom.dual
            transforms.append(t)
            dual_atoms.append(a)
        transform = afstack(transforms)
        nonsm = separable(transform.dual_shape, dual_atoms,
                          transform.dual_slices)
        return transform, nonsm

    def nonsmooth_objective(self, x, check_feasibility=False):
        out = 0.
        for atom in self.atoms:
            out += atom.nonsmooth_objective(x,
                                            check_feasibility=check_feasibility)
        return out

    default_solver = FISTA
    def proximal(self, y, lipschitz_P=1, prox_control=None):
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

        yL = lipschitz_P * y
        if not hasattr(self, 'dualopt'):
            self.dualp = self.dual_composite(yL, lipschitz_P=lipschitz_P)
            #Approximate Lipschitz constant
            transform, _ = self.dual
            self.dual_reference_lipschitz = 1.05*power_L(transform, debug=prox_control['debug'])
            self.dualopt = container.default_solver(self.dualp)
            self.dualopt.debug = prox_control['debug']

        self.dualopt.composite.smooth_multiplier = 1./lipschitz_P
        self.dualp.lipschitz = self.dual_reference_lipschitz / lipschitz_P

        self._dual_prox_center = yL
        history = self.dualopt.fit(**prox_control)
        if prox_control['return_objective_hist']:
            return self.primal_from_dual(y, self.dualopt.composite.coefs/lipschitz_P), history
        else:
            return self.primal_from_dual(y, self.dualopt.composite.coefs/lipschitz_P)

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

    def dual_composite(self, y, lipschitz_P=1, initial=None):
        """
        Return a problem instance of the dual
        prox problem with a given y value.
        """
        self._dual_prox_center = y
        transform, separable_atom = self.dual
        if initial is None:
            initial = np.random.standard_normal(transform.dual_shape)
        nonsmooth_objective = separable_atom.nonsmooth_objective
        prox = separable_atom.proximal
        return composite(self._dual_smooth_objective, nonsmooth_objective, prox, initial, 1./lipschitz_P)

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

        transform, separable_atom = self.dual
        prox = separable_atom.proximal
        nonsmooth_objective = separable_atom.nonsmooth_objective

        if initial is None:
            initial = np.random.standard_normal(transform.dual_shape)
        return composite(self.conjugate_smooth_objective, nonsmooth_objective, prox, initial, smooth_multiplier)
        

    def composite(self, smooth_multiplier=1., initial=None):
        """
        Create a composite object for solving the general problem with the two-loop algorithm
        """

        if initial is None:
            initial = np.random.standard_normal(self.atoms[0].primal_shape)
        prox = self.proximal
        nonsmooth_objective = self.nonsmooth_objective
        
        return composite(self.loss.smooth_objective, nonsmooth_objective, prox, initial, smooth_multiplier)

