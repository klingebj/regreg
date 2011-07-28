import numpy as np
from scipy import sparse
from algorithms import FISTA
from composite import (composite, nonsmooth as nonsmooth_composite,
                       smooth as smooth_composite)
from affine import (vstack as afvstack, identity as afidentity, power_L,
                    selector as afselector)
from separable import separable
#from conjugate import conjugate
from atoms import affine_atom as nonsmooth_affine_atom
from cones import zero_constraint, zero as zero_nonsmooth, affine_cone

class container(composite):
    """
    A container class for storing/combining seminorm_atom classes
    """
    def __init__(self, *atoms):
        self.nonsmooth_atoms = []
        self.smooth_atoms = []
        for atom in atoms:
            if (isinstance(atom, nonsmooth_composite) or 
                isinstance(atom, nonsmooth_affine_atom) or
                isinstance(atom, affine_cone)):
                self.nonsmooth_atoms.append(atom)
            elif isinstance(atom, smooth_composite):
                self.smooth_atoms.append(atom)
            else:
                raise ValueError('each atom should either be a smooth or nonsmooth composite')

        if len(self.nonsmooth_atoms) == 0 and len(self.smooth_atoms) == 0:
            raise ValueError('must specify some atoms')

        if len(self.nonsmooth_atoms) == 0:
            self.nonsmooth_atoms = [zero_nonsmooth(self.smooth_atoms[0].primal_shape)]
        transform, _ = self.dual
        self.coefs = np.zeros(transform.primal_shape)


    @property
    def dual(self):
        if not hasattr(self, "_dual"):
            transforms = []
            dual_atoms = []
            if len(self.nonsmooth_atoms) > 1:
                for atom in self.nonsmooth_atoms:
                    t, a = atom.dual
                    transforms.append(t)
                    dual_atoms.append(a)
                transform = afvstack(transforms)
                nonsm = separable(transform.dual_shape, dual_atoms,
                                  transform.dual_slices)
                self._dual = transform, nonsm
            else:
                self._dual = self.nonsmooth_atoms[0].dual
        return self._dual

    def smooth_objective(self, x, mode='both', check_feasibility=False):
        value, grad = 0, 0
        if mode == 'func':
            for atom in self.smooth_atoms:
                value += atom.smooth_objective(x, mode=mode, 
                                               check_feasibility=check_feasibility)
            return value
        elif mode == 'both':
            for atom in self.smooth_atoms:
                v, g = atom.smooth_objective(x, mode=mode, 
                                               check_feasibility=check_feasibility)
                value += v
                grad += g
            return value, grad

        elif mode == 'grad':
            for atom in self.smooth_atoms:
                grad += atom.smooth_objective(x, mode=mode, 
                                              check_feasibility=check_feasibility)
            return grad
        else:
            raise ValueError("Mode specified incorrectly")

    def nonsmooth_objective(self, x, check_feasibility=False):
        out = 0.
        for atom in self.nonsmooth_atoms:
            out += atom.nonsmooth_objective(x,
                                            check_feasibility=check_feasibility)
        return out

    default_solver = FISTA
    def proximal(self, y, lipschitz=1, prox_control=None):
        """
        The proximal function for the primal problem
        """

        transform, separable_atom = self.dual

        if not (isinstance(transform, afidentity) or
                isinstance(transform, afselector)):
            #Default fitting parameters
            prox_defaults = {'max_its': 5000,
                             'min_its': 5,
                             'return_objective_hist': False,
                             'tol': 1e-14,
                             'debug':False}

            if prox_control is not None:
                prox_defaults.update(prox_control)
            prox_control = prox_defaults

            yL = lipschitz * y
            if not hasattr(self, 'dualopt'):

                self._dual_response = yL
                initial = np.random.standard_normal(transform.dual_shape)
                nonsmooth_objective = separable_atom.nonsmooth_objective
                prox = separable_atom.proximal
                self.dualp = composite(self._dual_smooth_objective, nonsmooth_objective, prox, initial, 1./lipschitz)

                #Approximate Lipschitz constant
                self.dual_reference_lipschitz = 1.05*power_L(transform, debug=prox_control['debug'])
                self.dualopt = container.default_solver(self.dualp)
                self.dualopt.debug = prox_control['debug']

            self.dualopt.composite.smooth_multiplier = 1./lipschitz
            self.dualp.lipschitz = self.dual_reference_lipschitz / lipschitz

            self._dual_response = yL
            history = self.dualopt.fit(**prox_control)
            if prox_control['return_objective_hist']:
                return y - transform.adjoint_map(self.dualopt.composite.coefs/lipschitz), history
            else:
                return y - transform.adjoint_map(self.dualopt.composite.coefs/lipschitz)
        else:
            primal = separable_atom.conjugate
            return primal.proximal(y, lipschitz=lipschitz)

    def _dual_smooth_objective(self,v,mode='both', check_feasibility=False):

        """
        The smooth component and/or gradient of the dual objective        
        """
        
        # residual is the residual from the fit
        transform, _ = self.dual
        residual = self._dual_response - transform.adjoint_map(v)

        if mode == 'func':
            return (residual**2).sum() / 2.
        elif mode == 'both' or mode == 'grad':
            g = -transform.linear_map(residual)
            if mode == 'grad':
                return g
            if mode == 'both':
                return (residual**2).sum() / 2., g
        else:
            raise ValueError("Mode not specified correctly")

#     def conjugate_linear_term(self, u):
#         transform, _ = self.dual
#         return transform.adjoint_map(u)

#     def conjugate_primal_from_dual(self, u):
#         """
#         Calculate the primal coefficients from the dual coefficients
#         """
#         linear_term = self.conjugate_linear_term(-u)
#         return self.conjugate.smooth_objective(linear_term, mode='grad')


#     def conjugate_smooth_objective(self, u, mode='both', check_feasibility=False):
#         linear_term = self.conjugate_linear_term(u)
#         # XXX dtype manipulations -- would be nice not to have to do this
#         u = u.view(self.dual_dtype).reshape(())
#         if mode == 'both':
#             v, g = self.conjugate.smooth_objective(-linear_term, mode='both')
#             grad = np.empty((), self.dual_dtype)
#             for dual_atom, segment in zip(self.dual_atoms, self.dual_segments):
#                 transform, _ = dual_atom
#                 grad[segment] = -transform.linear_map(g)
#             # XXX dtype manipulations -- would be nice not to have to do this
#             return v, grad.reshape((1,)).view(np.float) 
#         elif mode == 'grad':
#             g = self.conjugate.smooth_objective(-linear_term, mode='grad')
#             grad = np.empty((), self.dual_dtype)
#             for dual_atom, segment in zip(self.dual_atoms, self.dual_segments):
#                 transform, _ = dual_atom
#                 grad[segment] = -transform.linear_map(g)
#             # XXX dtype manipulations -- would be nice not to have to do this
#             return grad.reshape((1,)).view(np.float) 
#         elif mode == 'func':
#             v = self.conjugate.smooth_objective(-linear_term, mode='func')
#             return v
#         else:
#             raise ValueError("mode incorrectly specified")

#     def conjugate_composite(self, conj=None, initial=None, smooth_multiplier=1., conjugate_tol=1.0e-08, epsilon=0.):
#         """
#         Create a composite object for solving the conjugate problem
#         """

#         if conj is not None:
#             self.conjugate = conj
#         if not hasattr(self, 'conjugate'):
#             #If the conjugate of the loss function is not provided use the generic solver
#             self.conjugate = conjugate(self.loss, tol=conjugate_tol,
#                                        epsilon=epsilon)

#         transform, separable_atom = self.dual
#         prox = separable_atom.proximal
#         nonsmooth_objective = separable_atom.nonsmooth_objective

#         if initial is None:
#             initial = np.random.standard_normal(transform.dual_shape)
#         return composite(self.conjugate_smooth_objective, nonsmooth_objective, prox, initial, smooth_multiplier)
        
