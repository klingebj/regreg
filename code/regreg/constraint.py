import numpy as np
from scipy import sparse
from algorithms import FISTA
from problem import dummy_problem
from atoms import primal_dual_seminorm_pairs

#XXX this could be instantiated with just smooth_eval instead of something having 
# smooth_eval method...

class constraint(object):
    """
    A constraint container class for storing/combining constraints
    specified in terms of the conjugate of a smooth convex function
    and some atoms. 

    This class solves the problem

    .. math::

       \text{minimize}_{u_i} {\cal L}^*(-\sum_i D_i^Tu_i) + \sum_i (u_i'\alpha_i + \lambda_i h_i(u_i)

    where :math:`h_i` are the seminorms dual to the atoms.

    The conjugate should be smooth
    """
    def __init__(self, conjugate, *atoms):
        self.conjugate = conjugate
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
            # the dual_seminorm property has this multiplier 1/atom.l by default
            # because the dual_seminorm is the support function of the polar
            # body so it should be divided by l if l!=1

            # for the constrained problem, atom.l represents the
            # value in the constraint -- in the dual this becomes
            # a multiplier of a dual_seminorm, so here we change it to l
            dual_atom.l = atom.l 
            print atom.l, dual_atom.l, 'constraint'
            self.dual_atoms.append(dual_atom)

            self.dual_shapes += [atom.dual_shape]
        self.dual_dtype = np.dtype([('dual_%d' % i, np.float, shape) 
                                    for i, shape in enumerate(self.dual_shapes)])
        self.dual_segments = self.dual_dtype.names 

    def linear_term(self, u):
        lterm = 0
        # XXX dtype manipulations -- would be nice not to have to do this
        u = u.view(self.dual_dtype).reshape(())
        for atom, segment in zip(self.atoms, self.dual_segments):
            lterm += atom.adjoint_map(u[segment])
        return lterm

    def smooth_eval(self, u, mode='both'):
        linear_term = self.linear_term(u)
        # XXX dtype manipulations -- would be nice not to have to do this
        u = u.view(self.dual_dtype).reshape(())
        if mode == 'both':
            v, g = self.conjugate.smooth_eval(-linear_term, mode='both')
            grad = np.empty((), self.dual_dtype)
            for atom, segment in zip(self.atoms, self.dual_segments):
                grad[segment] = -atom.affine_map(g)
                v -= atom.affine_objective(u[segment])
            # XXX dtype manipulations -- would be nice not to have to do this
            return v, grad.reshape((1,)).view(np.float) 
        elif mode == 'grad':
            g = self.conjugate.smooth_eval(-linear_term, mode='grad')
            grad = np.empty((), self.dual_dtype)
            for atom, segment in zip(self.atoms, self.dual_segments):
                grad[segment] = -atom.affine_map(g)
            # XXX dtype manipulations -- would be nice not to have to do this
            return grad.reshape((1,)).view(np.float) 
        elif mode == 'func':
            v = self.conjugate.smooth_eval(-linear_term, mode='func')
            for atom, segment in zip(self.atoms, self.dual_segments):
                v -= atom.affine_objective(u[segment])
            return v
        else:
            raise ValueError("mode incorrectly specified")


    def __add__(self, other):
        #Combine two constraints
        def atoms():
            for obj in [self, other]:
                for atom in obj.atoms:
                    yield atom
        new_conjugate = smooth_function(other.conjugate, self.conjugate)
        return constraint(new_conjugate, *atoms())

    def evaluate_constraint(self, x):
        out = 0.
        for atom in self.atoms:
            out += atom.evaluate_constraint(x)
        return out
    
    def evaluate_dual_seminorm(self, u):
        out = 0.
        # XXX dtype manipulations -- would be nice not to have to do this
        u = u.view(self.dual_dtype).reshape(())
        for dual_atom, segment in zip(self.dual_atoms, self.dual_segments):
            out += dual_atom.evaluate_seminorm(u[segment])
        return out
    
    def dual_prox(self, u, L_P=1, with_history=False, debug=False, max_its=5000, tol=1e-14):
        """
        Return (unique) minimizer

        .. math::

           v^{\lambda}(u) = \text{argmin}_{v \in \real^m} \frac{L}{2}
           \|v-u\|^2_2 + \sum_i \lambda_i h_i(v)

        where *m*=u.shape[0]=np.sum(self.dual_dims), :math:`M`=self.M
        and :math:`h_i` is the seminorm of 
        self.dual_atoms[i] and 
        :math:`\lambda_i`=self.atoms[i].l.

        """
        value = np.empty((), self.dual_dtype)
        # XXX dtype manipulations -- would be nice not to have to do this
        u = u.view(self.dual_dtype).reshape(())
        for dual_atom, segment in zip(self.dual_atoms, self.dual_segments):
            value[segment] = dual_atom.primal_prox(u[segment], L_P)
        return value.reshape((1,)).view(np.float)

    def primal_from_dual(self, u):
        """
        Calculate the primal coefficients from the dual coefficients
        """
        linear_term = self.linear_term(-u)
        return self.conjugate.smooth_eval(linear_term, mode='grad')

    def dual_problem(self, smooth_multiplier=1., initial=None):
        prox = self.dual_prox
        nonsmooth = self.evaluate_dual_seminorm
        smooth_eval = self.smooth_eval
        if initial is None:
            initial = np.empty((1,), self.dual_dtype).view(np.float)
            initial[:] = np.random.standard_normal(initial.shape)
        if nonsmooth(initial) + smooth_eval(initial,mode='func') == np.inf:
            raise ValueError('initial point is not feasible')
        return dummy_problem(smooth_eval, nonsmooth, prox, initial, smooth_multiplier)
