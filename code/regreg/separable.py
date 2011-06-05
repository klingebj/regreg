"""
This module implements the notion of a separable support function / constraint
regularizer or penalty.

The penalty is specified by a primal shape, a sequence of atoms and
a sequence of slicing objects.
"""
from atoms import seminorm_atom, affine_atom
from affine import selector

class separable(object):

    def __init__(self, primal_shape, atoms, groups):
        self.primal_shape = primal_shape
        self.atoms = atoms
        self.groups = groups
        self.selector_atoms = [affine_atom(atom, selector(group, primal_shape))
                               for atom, group in zip(self.atoms,
                                                      self.groups)]

        self.dtype = np.dtype([('group_%d' % i, np.float, atom.primal_shape) 
                               for i, atom in enumerate(self.atoms)])

