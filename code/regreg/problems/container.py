import numpy as np
from scipy import sparse

from ..algorithms import FISTA
from ..problems.composite import (composite, nonsmooth as nonsmooth_composite,
                        smooth as smooth_composite)
from ..affine import (vstack as afvstack, identity as afidentity, power_L,
                     selector as afselector)
from ..problems.separable import separable
from ..problems.dual_problem import dual_problem, stacked_dual
from ..atoms import affine_atom as nonsmooth_affine_atom
from ..atoms.cones import zero_constraint, zero as zero_nonsmooth, affine_cone

from ..identity_quadratic import identity_quadratic

class container(composite):
    """
    A container class for storing/combining seminorm_atom classes

    Notes
    -----

    This function copies the nonsmooth atoms
    """

    def __init__(self, *atoms, **keywords):
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
            self.nonsmooth_atoms = [zero_nonsmooth(self.smooth_atoms[0].shape)]

        self.transform, self.atom = stacked_dual(self.smooth_atoms[0].shape, *self.nonsmooth_atoms)
        self.coefs = np.zeros(self.transform.primal_shape)

        # add up all the smooth_atom quadratics
        # to be added to nonsmoooth_objective

        self.smoothq = identity_quadratic(0,0,0,0)
        for atom in self.smooth_atoms:
            self.smoothq = self.smoothq + atom.quadratic

    def smooth_objective(self, x, mode='both', check_feasibility=False):
        """
        The smooth_objective DOES NOT INCLUDE the identity
        quadratic of all the smooth atoms.
        """
        value, grad = 0, np.zeros(x.shape)
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
        # now add the smooth_atoms' quadratic
        out += self.smoothq.objective(x, 'func')
        out += self.quadratic.objective(x, 'func')
        return out

    default_solver = FISTA
    def proximal(self, proxq, prox_control=None):
        """
        The proximal function for the primal problem
        """

        transform, atom = self.transform, self.atom

        if not (isinstance(transform, afidentity) or
                isinstance(transform, afselector)):
            #Default fitting parameters
            prox_defaults = {'max_its': 5000,
                             'min_its': 5,
                             'return_objective_hist': False,
                             'tol': 1e-14,
                             'debug':False,
                             'backtrack':False}

            if prox_control is not None:
                prox_defaults.update(prox_control)
            prox_control = prox_defaults

            primal_objective = zero_nonsmooth(transform.primal_shape)
            primal_objective.quadratic = proxq + self.smoothq + self.quadratic
            dual_objective = primal_objective.conjugate
            dualp = dual_problem(dual_objective,
                                 transform,
                                 atom)

            #Approximate Lipschitz constant
            if not 'dual_reference_lipschitz' in prox_control.keys():
                # shouldn't do this over and over
                self.dual_reference_lipschitz = 1.05*power_L(transform, debug=prox_control['debug'])
            else:
                self.dual_reference_lipschitz = prox_control['dual_reference_lipschitz']
                prox_control.pop('dual_reference_lipschitz')
                
            dualopt = container.default_solver(dualp)
            dualopt.debug = prox_control['debug']

            if prox_control['backtrack']:
                #If backtracking set start_inv_step
                prox_control['start_inv_step'] = self.dual_reference_lipschitz / proxq.coef

            # the lipschitz estimate comes from the
            # fact that the conjugate of a quadratic
            # with coef lipschitz is quadratic with coef 1/lipschitz

            dualp.lipschitz = self.dual_reference_lipschitz / proxq.coef

            if hasattr(self, 'dual_minimizer'):
                dualopt.composite.coefs[:] = self.dual_minimizer
            history = dualopt.fit(**prox_control)
            dual_minimizer = dualopt.composite.coefs
            lipschitz, x, grad = proxq.coef, proxq.center, proxq.linear_term
            if prox_control['return_objective_hist']:
                return x - grad / lipschitz - transform.adjoint_map(dualopt.composite.coefs/lipschitz), history
            else:
                return x - grad / lipschitz - transform.adjoint_map(dualopt.composite.coefs/lipschitz)

        else:
            primal = atom.conjugate
            if isinstance(transform, afselector):
                z = x.copy()
                z[transform.index_obj] = primal.proximal(proxq[transform.index_obj])
                return z
            else:
                return primal.proximal(proxq)

    def solve(self, quadratic=None, return_optimum=False, **fit_args):
        if quadratic is not None:
            oldq, newq = self.quadratic, self.quadratic + quadratic
        else:
            oldq = newq = self.quadratic

        solver = FISTA(self)
        solver.fit(**fit_args)

        if return_optimum:
            value = (self.objective(self.coefs), self.coefs)
        else:
            value = self.coefs
        self.quadratic = oldq
        return value

