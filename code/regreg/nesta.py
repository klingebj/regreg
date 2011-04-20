import seminorm, smooth
import numpy as np

class smoothed_seminorm(smooth.smooth_function):

    """
    Given a seminorm :math:`h_K(D\beta)`, this
    class creates a smoothed version

    .. math::

        h_{K,\varepsilon}(D\beta) = \sup_{u \in K}u'D\beta - \frac{\epsilon}{2}
        \|u\|^2_2

    The objective value is given by

    .. math::

       h_{K,\varepsilon}(D\beta) = \frac{1}{2\epsilon} \|D\beta\|^2_2- \frac{\epsilon}{2} \|D\beta/\epsilon - P_K(D\beta/\epsilon)\|^2_2

    and the gradient is given by

    .. math::

       \nabla_{\beta} h_{K,\varepsilon}(D\beta) = D'P_K(D\beta/\epsilon)

    If a seminorm has several atoms, then :math:`D` is a
    `stacked' version and :math:`K` is a product
    of corresponding convex sets.
    
    """
    def __init__(self, semi, epsilon=0.01):
        self.seminorm = semi
        self.epsilon = epsilon
        
    def smooth_eval(self, beta, mode='both'):
        """
        Evaluate a smooth function and/or its gradient

        if mode == 'both', return both function value and gradient
        if mode == 'grad', return only the gradient
        if mode == 'func', return only the function value
        """

        if mode == 'both':
            objective, grad = 0, 0
            for atom in self.seminorm.atoms:
                u = atom.multiply_by_D(beta)
                ueps = u / self.epsilon
                projected_ueps = atom.dual_prox(ueps)
                objective += ((u**2).sum() / (2. * self.epsilon) - self.epsilon / 2. *
                        ((ueps - projected_ueps)**2).sum())
                grad += atom.multiply_by_DT(projected_ueps)
                return objective, grad
        elif mode == 'grad':
            grad = 0
            for atom in self.seminorm.atoms:
                u = atom.multiply_by_D(beta)
                ueps = u / self.epsilon
                projected_ueps = atom.dual_prox(ueps)
                grad += atom.multiply_by_DT(projected_ueps)
            return grad
        elif mode == 'func':
            objective = 0
            for atom in self.seminorm.atoms:
                u = atom.multiply_by_D(beta)
                ueps = u / self.epsilon
                projected_ueps = atom.dual_prox(ueps)
                objective += ((u**2).sum() / (2. * self.epsilon) - self.epsilon / 2. *
                        ((ueps - projected_ueps)**2).sum())
            return objective
        else:
            raise ValueError("mode incorrectly specified")

class sum(smooth.smooth_function):

    def __init__(self, multipliers_function_tuples):
        self.multiplier_function_tuples = multipliers_function_tuples

    def smooth_eval(self, beta, mode='both'):
        """
        Evaluate a smooth function and/or its gradient

        if mode == 'both', return both function value and gradient
        if mode == 'grad', return only the gradient
        if mode == 'func', return only the function value
        """

        if mode == 'both':
            objective, grad = 0, 0
            for multiplier, smooth_f in self.multiplier_function_tuples:
                obj_f, grad_f = smooth_f(beta, mode='both')
                objective += multiplier * obj_f
                grad += multiplier * grad_f
            return objective, grad
        elif mode == 'grad':
            grad = 0
            for multiplier, smooth_f in self.multiplier_function_tuples:
                grad_f = smooth_f(beta, mode='grad')
                grad += multiplier * grad_f
            return grad
        elif mode == 'func':
            objective = 0
            for multiplier, smooth_f in self.multiplier_function_tuples:
                obj_f = smooth_f(beta, mode='func')
                objective += multiplier * obj_f
            return objective
        else:
            raise ValueError("mode incorrectly specified")


if __name__ == "__main__":
    import atoms
    sparsity = atoms.l1norm(50)
    huber = smoothed_seminorm(sparsity)

    
