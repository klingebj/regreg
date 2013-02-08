"""
This module implements the notion of a separable support function / constraint
regularizer or penalty.

The penalty is specified by a primal shape, a sequence of atoms and
a sequence of slicing objects.
"""

import numpy as np

from .affine import selector
from .atoms import atom
from .simple import simple_problem
from .cones import zero

def has_overlap(shape, groups):
    """
    Determine whether the groups, viewed as slices of an array
    with given shape, have any overlap.

    Parameters
    ----------
    shape : tuple
        A tuple of integers representing a shape for an array.
    groups : sequence
        A sequence of objects that can be viewed as slices of
        an ndarray with shape==shape.

    Returns
    -------
    res : boolean
        True if the slices overlap, else False.

    Examples
    --------
    >>> has_overlap((4,5), [slice(2,3), slice(4,5)])
    False
    >>> has_overlap((4,5), [slice(2,3), [Ellipsis, slice(4,5)]])
    True
    """
    indices = np.arange(np.product(shape)).reshape(shape)
    subsets = []
    for group in groups:
        subsets.append(set(list(indices[group].reshape(-1))))
    subsets = tuple(subsets)
    for i in range(len(subsets)):
        for j in range(i):
            subset1 = subsets[i]
            subset2 = subsets[j]
            if subset1 != subset2 and subset1.intersection(subset2):
                    return True
    return False

class separable(atom):

    def __init__(self, shape, atoms, groups, test_for_overlap=False,
                 initial=None):
        if test_for_overlap and has_overlap(shape, groups):
            raise ValueError('groups are not separable')
        self.primal_shape = self.dual_shape = shape
        self.groups = groups
        self.atoms = atoms
        self.zero_atom = zero(shape)

        if initial is None:
            self.coefs = np.zeros(shape)

    def seminorm(self, x, lagrange=None, check_feasibility=False):
        value = 0.
        for atom, group in zip(self.atoms, self.groups):
            value += atom.seminorm(x[group], lagrange=lagrange, check_feasibility=check_feasibility)
        return value

    def constraint(self, x, bound=None):
        value = 0.
        for atom, group in zip(self.atoms, self.groups):
            value += atom.constraint(x[group], bound=bound)
        return value

    def nonsmooth_objective(self, x, check_feasibility=False):
        value = 0
        for atom, group in zip(self.atoms, self.groups):
            value += atom.nonsmooth_objective(x[group], check_feasibility=check_feasibility)
        return value

    def proximal(self, proxq, prox_control=None):
        # This allows separable to not have every coefficient penalized in a natural way
        # because this instantiates it at the prox map of the zero penalty
        v = self.zero_atom.proximal(proxq)
        for atom, group in zip(self.atoms, self.groups):
            v[group] = atom.proximal(proxq[group], prox_control)
        return v

    @property
    def conjugate(self):
        penalty = separable(self.primal_shape,
                            [atom.conjugate for atom in self.atoms],
                            self.groups)
        return penalty

    def __repr__(self):
        return "separable(%s, %s, %s)" % (`self.primal_shape`,
                                          `self.atoms`,
                                          `self.groups`)

    @property
    def selectors(self):
        return [selector(group, self.shape) for group in self.groups]

class separable_problem(simple_problem):
    
    def __init__(self, smooth_atom, shape, atoms, groups, test_for_overlap=False):
        nonsmooth_atom = separable(shape, atoms, groups, test_for_overlap)
        simple_problem.__init__(self, smooth_atom, nonsmooth_atom)
        
    @staticmethod
    def singleton(atom, smooth_f):
        return separable_problem(smooth_f, atom.primal_shape, [atom], [slice(None)], False)

    @staticmethod
    def fromatom(separable_atom, smooth_f):
        return separable_problem(smooth_f, separable_atom.primal_shape, 
                                 separable_atom.atoms,
                                 separable_atom.groups, False)
    @property
    def selectors(self):
        return [selector(group, self.nonsmooth_atom.primal_shape) for group in self.nonsmooth_atom.groups]
