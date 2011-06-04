import numpy as np
from container import container
from smooth import smooth_function, zero

class smoothed_seminorm(smooth_function):

    def __init__(self, atoms, epsilon=0.1, prox_center=None,
                 store_argmin=False):
        """
        Given a seminorm :math:`h_K(D\beta)`, this
        class creates a smoothed version

        .. math::

            h_{K,\varepsilon}(D\beta+\alpha) = \sup_{u \in K}u'(D\beta+\alpha) - \frac{\epsilon}{2}
            \|u-u_0\|^2_2

        The point :math:`u_0` is a prox center around which we wish
        to smooth the seminorm. It is a set of dual variables.
        The objective value is given by

        .. math::

           h_{K,\varepsilon}(D\beta+\alpha) = u_0'(D\beta+\alpha) + \frac{1}{2\epsilon} \|D\beta+\alpha\|^2_2- \frac{\epsilon}{2} \|u_0+(D\beta+\alpha)/\epsilon - P_K(u_0+(D\beta+\alpha)/\epsilon)\|^2_2

        and the gradient is given by the maximizer

        .. math::

           \nabla_{\beta} h_{K,\varepsilon}(D\beta+\alpha) = D'P_K(u_0+(D\beta+\alpha)/\epsilon)

        If a seminorm has several atoms, then :math:`D` is a
        `stacked' version and :math:`K` is a product
        of corresponding convex sets.

        """
        self.epsilon = epsilon
        if not np.all([(not atom.constraint for atom in atoms)]):
            raise ValueError('all atoms should be in Lagrange form, i.e. bound=None')
        if self.epsilon <= 0:
            raise ValueError('to smooth, epsilon must be positive')
        self.primal_shape = atoms[0].primal_shape
        try:
            for atom in atoms:
                assert(atom.primal_shape == self.primal_shape)
        except:
            raise ValueError("Atoms have different primal shapes")
        self.coefs = np.zeros(self.primal_shape)
        zero_sm = zero(self.primal_shape)
        self.dual_dtype = container(zero, *atoms).dual_dtype
        self.atoms = atoms

        if prox_center is not None:
            # XXX dtype manipulations -- would be nice not to have to do this
            self.prox_center = prox_center.view(self.dual_dtype).reshape(())
        else:
            self.prox_center = None

        # for NESTA the argmin corresponds to a feasible
        # point for the dual constraint

        self.store_argmin = store_argmin


    def smooth_eval(self, beta, mode='both'):
        """
        Evaluate a smooth function and/or its gradient

        if mode == 'both', return both function value and gradient
        if mode == 'grad', return only the gradient
        if mode == 'func', return only the function value
        """

        if mode == 'both':
            objective, grad = 0, 0
            for i, atom in enumerate(self.atoms):
                u = atom.affine_map(beta)
                ueps = u / self.epsilon
                if self.prox_center is not None:
                    prox = self.prox_center['dual_%d' % i]
                    argmin, optimal_value = atom.dual_prox_optimum(prox+ueps, self.epsilon)                    
                    if self.store_argmin:
                        self.argmin = argmin
                    objective += self.epsilon / 2. * np.linalg.norm(ueps)**2 - optimal_value + (prox*u).sum()
                else:
                    argmin, optimal_value = atom.dual_prox_optimum(ueps, self.epsilon)                    
                    objective += self.epsilon / 2. * np.linalg.norm(ueps)**2 - optimal_value

                grad += atom.adjoint_map(argmin)
                if self.store_argmin:
                    self.argmin = argmin

            return objective, grad
        elif mode == 'grad':
            grad = 0
            for i, atom in enumerate(self.atoms):
                u = atom.affine_map(beta)
                ueps = u / self.epsilon
                if self.prox_center is not None:
                    prox = self.prox_center['dual_%d' % i]
                    argmin = atom.dual_prox(prox+ueps, self.epsilon)         
                else:
                    argmin = atom.dual_prox(ueps, self.epsilon)             
                grad += atom.adjoint_map(argmin)
                if self.store_argmin:
                    self.argmin = argmin

            return grad 
        elif mode == 'func':
            objective = 0
            for i, atom in enumerate(self.atoms):
                u = atom.affine_map(beta)
                ueps = u / self.epsilon
                if self.prox_center is not None:
                    prox = self.prox_center['dual_%d' % i]
                    _, optimal_value = atom.dual_prox_optimum(prox+ueps, self.epsilon)                    
                    objective += self.epsilon / 2. * np.linalg.norm(ueps)**2 - optimal_value + (prox*u).sum()
                else:
                    _, optimal_value = atom.dual_prox_optimum(ueps, self.epsilon)                    
                    objective += self.epsilon / 2. * np.linalg.norm(ueps)**2 - optimal_value
            return objective 
        else:
            raise ValueError("mode incorrectly specified")


class smoothed_constraint(smooth_function):

    def __init__(self, atom, epsilon=0.1, prox_center=None,
                 store_argmin=False):
        """
        Given a constraint :math:`\delta_K(\beta+\alpha)=h_K^*(\beta)`,
        that is, a possibly atom whose linear_operator is None, and
        whose offset is :math:`\alpha` this
        class creates a smoothed version

        .. math::

            \delta_{K,\varepsilon}(\beta+\alpha) = \sup_{u}u'(\beta+\alpha) - \frac{\epsilon}{2} \|u-u_0\|^2_2 - h_K(u)

        The objective value is given by

        .. math::

           \delta_{K,\varepsilon}(\beta) = \beta'u_0 + \frac{1}{2\epsilon} \|\beta\|^2_2- \frac{\epsilon}{2} \left(\|P_K(u_0+(\beta+\alpha)/\epsilon)\|^2_2 + h_K\left(u_0+(\beta+\alpha)/\epsilon - P_K(u_0+(\beta+\alpha)/\epsilon)\right)

        and the gradient is given by the maximizer above

        .. math::

           \nabla_{\beta} \delta_{K,\varepsilon}(\beta+\alpha) = u_0+(\beta+\alpha)/\epsilon - P_K(u_0+(\beta+\alpha)/\epsilon)

        If a seminorm has several atoms, then :math:`D` is a
        `stacked' version and :math:`K` is a product
        of corresponding convex sets.

        """
        self.epsilon = epsilon
        if atom.bound is None:
            raise ValueError('atom should be in constraint form, i.e. bound != None')
        if self.epsilon <= 0:
            raise ValueError('to smooth, epsilon must be positive')
        self.primal_shape = atom.primal_shape
        self.coefs = np.zeros(self.primal_shape)
        zero_sm = zero(self.primal_shape)
        self.dual_dtype = container(zero, atom).dual_dtype
        self.atom = atom
        self.dual_atom = atom.conjugate

        if prox_center is not None:
            # XXX dtype manipulations -- would be nice not to have to do this
            self.prox_center = prox_center.view(self.dual_dtype).reshape(())
        else:
            self.prox_center = None

        # for TFOCS the argmin corresponds to the 
        # primal solution because the smoothed_constraint
        # is the smooth objective of the dual problem

        self.store_argmin = store_argmin

    def smooth_eval(self, beta, mode='both'):
        """
        Evaluate a smooth function and/or its gradient

        if mode == 'both', return both function value and gradient
        if mode == 'grad', return only the gradient
        if mode == 'func', return only the function value
        """

        atom, dual_atom = self.atom, self.dual_atom
        if self.prox_center is not None:
            prox = self.prox_center['dual_0']
        else:
            prox = None
        if mode == 'both':
            u = atom.affine_map(beta)
            ueps = u / self.epsilon
            if prox is not None:
                argmin, optimal_value = dual_atom.primal_prox_optimum(prox+ueps, self.epsilon)                    
                objective = self.epsilon / 2. * np.linalg.norm(ueps)**2 - optimal_value + (prox*u).sum()
            else:
                argmin, optimal_value = dual_atom.primal_prox_optimum(ueps, self.epsilon)                    
                objective = self.epsilon / 2. * np.linalg.norm(ueps)**2 - optimal_value
            grad = atom.adjoint_map(argmin)
            if self.store_argmin:
                self.argmin = argmin

            return objective, grad
        elif mode == 'grad':
            grad = 0
            u = atom.affine_map(beta)
            ueps = u / self.epsilon
            if prox is not None:
                argmin = dual_atom.primal_prox(ueps+prox, self.epsilon)     
            else:
                argmin = dual_atom.primal_prox(ueps, self.epsilon)         
            grad = atom.adjoint_map(argmin)

            if self.store_argmin:
                self.argmin = argmin
            return grad 
        elif mode == 'func':
            objective = 0
            u = atom.affine_map(beta)
            ueps = u / self.epsilon
            if prox is not None:
                _, optimal_value = dual_atom.primal_prox_optimum(ueps+prox, self.epsilon)                    
                objective = self.epsilon / 2. * np.linalg.norm(ueps)**2 - optimal_value + (prox*u).sum()
            else:
                _, optimal_value = dual_atom.primal_prox_optimum(ueps, self.epsilon)                    
                objective = self.epsilon / 2. * np.linalg.norm(ueps)**2 - optimal_value
            return objective 
        else:
            raise ValueError("mode incorrectly specified")
