from numpy.linalg import norm
from numpy import zeros

# local import

from identity_quadratic import identity_quadratic

class composite(object):
    """
    A generic way to specify a problem in composite form.
    """

    def __init__(self, smooth_objective, initial, smooth_multiplier=1, lipschitz=None, 
                 quadratic_spec=(None,None,None,0)):

        self.coefs = initial.copy()
        self._smooth_objective = smooth_objective
        self.proximal = proximal
        self.smooth_multiplier = smooth_multiplier
        self._lipschitz = lipschitz

        self.quadratic_spec = quadratic_spec

    def nonsmooth_objective(self, x, check_feasibility=False):
        if self.quadratic is not None:
            return self.quadratic.objective(x, 'func')
        return 0

    def smooth_objective(self, x, mode='both', check_feasibility=False):
        '''
        The smooth_objective and the quadratic_objective combined.
        '''
        smooth_output = self._smooth_objective(x, mode=mode, check_feasibility=check_feasibility)

        if self.smooth_multiplier != 1:
            if mode == 'both':
                smooth_output = (self.smooth_multiplier * smooth_output[0], 
                                 self.smooth_multiplier * smooth_output[1])
            elif mode == 'grad' or mode == 'func':
                smooth_output = self.smooth_multiplier * smooth_output
            else:
                raise ValueError("Mode incorrectly specified")

        return smooth_output

    def objective(self, x, check_feasibility=False):
        return self.smooth_objective(x,mode='func', check_feasibility=check_feasibility) + self.nonsmooth_objective(x, check_feasibility=check_feasibility)

    def proximal_optimum(self, lipschitz, x, grad):
        """
        Returns

        .. math::

           \inf_{v \in \mathbb{R}^p} \frac{L}{2}
           \|x-v\|^2_2 + \lambda h(v)

        where *p*=x.shape[0] and :math:`h(v)` = self.seminorm(v).

        Here, h represents the nonsmooth part and the quadratic
        part of the composite object.

        """
        argmin = self.proximal(lipschitz, x, grad)
        if self.quadratic is None:
            return argmin, lipschitz * norm(x-argmin)**2 / 2. + self.nonsmooth_objective(argmin)  
        else:
            return argmin, lipschitz * norm(x-argmin)**2 / 2. + self.nonsmooth_objective(argmin) + self.quadratic.objective(argmin, 'func') 

    def proximal_step(self, lipschitz, x, grad, prox_control=None):
        """
        Compute the proximal optimization

        prox_control: If not None, then a dictionary of parameters for the prox procedure
        """
        if prox_control is None:
            return self.proximal(lipschitz, x, grad)
        else:
            return self.proximal(lipschitz, x, grad, prox_control=prox_control)

        
    def set_quadratic(self, coef, offset, linear_term, constant_term):
        self._quadratic = identity_quadratic(coef, offset, linear_term, constant_term)

    def get_quadratic(self):
        if hasattr(self, "_quadratic") and self._quadratic.anything_to_return:
            return self._quadratic
        else:
            return None
    quadratic = property(get_quadratic)

    def get_lipschitz(self):
        if self.quadratic is not None and self.quadratic.coef is not None:
            return self._lipschitz + self.quadratic.coef
        return self._lipschitz

    def set_lipschitz(self, value):
        if value < 0:
            raise ValueError('Lipschitz constant must be non-negative')
        self._lipschitz = value
    lipschitz = property(get_lipschitz, set_lipschitz)

class nonsmooth(composite):
    """
    A composite subclass that explicitly returns 0
    as smooth_objective.
    """

    smooth_multiplier = 1

    def smooth_objective(self, x, mode='both', check_feasibility=False):
        if mode == 'both':
            return 0., zeros(x.shape)
        elif mode == 'func':
            return 0.
        elif mode == 'grad':
            return zeros(x.shape)
        raise ValueError("Mode not specified correctly")


class smooth(composite):

    """
    A composite subclass that has 0 as 
    nonsmooth_objective and the proximal
    is a null-op.
    """

    def __init__(self, smooth_objective, initial, smooth_multiplier=1, lipschitz=None):
        composite.__init__(self, smooth_objective, 
                           initial,
                           smooth_multiplier=smooth_multiplier,
                           lipschitz=lipschitz)

    def proximal(self, lipschitz, x, grad):
        if self.quadratic is None:
            return x
        else:
            proxq = identity_quadratic(lipschitz, -x, grad, 0)
            totalq = self.quadratic + proxq
            return -totalq.linear_term / totalq.coef

# This can now be done with a method of the atom 
class smoothed(smooth):

    def __init__(self, atom, epsilon=0.1,
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
        import warnings
        warnings.warn('to be deprecated, use the smoothed method of atom instead')
        self.epsilon = epsilon
        if self.epsilon <= 0:
            raise ValueError('to smooth, epsilon must be positive')
        self.primal_shape = atom.primal_shape

        self.dual = atom.dual

        # for TFOCS the argmin corresponds to the 
        # primal solution 

        self.store_argmin = store_argmin

    def smooth_objective(self, beta, mode='both', check_feasibility=False):
        """
        Evaluate a smooth function and/or its gradient

        if mode == 'both', return both function value and gradient
        if mode == 'grad', return only the gradient
        if mode == 'func', return only the function value
        """

        linear_transform, dual_atom = self.dual
        constant_term = dual_atom.constant_term

        u = linear_transform.linear_map(beta)
        ueps = u / self.epsilon
        if mode == 'both':
            argmin, optimal_value = dual_atom.proximal_optimum(self.epsilon, ueps, 0)                    
            objective = self.epsilon / 2. * norm(ueps)**2 - optimal_value + constant_term
            grad = linear_transform.adjoint_map(argmin)
            if self.store_argmin:
                self.argmin = argmin
            return objective, grad
        elif mode == 'grad':
            argmin = dual_atom.proximal(self.epsilon, ueps, 0)
            grad = linear_transform.adjoint_map(argmin)
            if self.store_argmin:
                self.argmin = argmin
            return grad 
        elif mode == 'func':
            _, optimal_value = dual_atom.proximal_optimum(self.epsilon, ueps, 0)                    
            objective = self.epsilon / 2. * norm(ueps)**2 - optimal_value + constant_term
            return objective
        else:
            raise ValueError("mode incorrectly specified")


