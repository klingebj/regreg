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
        self.dual_atoms = [atom.conjugate for atom in self.atoms]
        self.groups = groups
        self.selector_atoms = [affine_atom(atom, selector(group, primal_shape))
                               for atom, group in zip(self.atoms,
                                                      self.groups)]

        self.dtype = np.dtype([('group_%d' % i, np.float, atom.primal_shape) 
                               for i, atom in enumerate(self.atoms)])

    def nonsmooth(self, u):
        out = 0.
        # XXX dtype manipulations -- would be nice not to have to do this
        u = u.view(self.dtype).reshape(())

        for atom, group in zip(self.atoms, self.dtype.names):
            out += atom.nonsmooth(u[segment])
        return out

    def prox(self, u, lipshitz_D=1.):
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
        for atom, segment in zip(self.atoms, self.dual_segments):
            v[segment] = atom.prox(u[segment], lipshitz_D)
        return v.reshape((1,)).view(np.float)
