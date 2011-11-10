from numpy.linalg import norm
from numpy import zeros

class composite(object):
    """
    A generic way to specify a problem in composite form.
    """

    def __init__(self, smooth_objective, nonsmooth_objective, proximal, initial, smooth_multiplier=1, lipschitz=None, compute_difference=True):
        self.coefs = initial.copy()
        self.nonsmooth_objective = nonsmooth_objective
        self._smooth_objective = smooth_objective
        self.proximal = proximal
        self.smooth_multiplier = smooth_multiplier
        self._lipschitz = lipschitz
        self.compute_difference = compute_difference

    def smooth_objective(self, x, mode='both', check_feasibility=False):
        output = self._smooth_objective(x, mode=mode, check_feasibility=check_feasibility)
        if self.smooth_multiplier == 1:
            return output
        else:
            if mode == 'both':
                return self.smooth_multiplier * output[0], self.smooth_multiplier * output[1]
            elif mode == 'grad' or mode == 'func':
                return self.smooth_multiplier * output
            else:
                raise ValueError("Mode incorrectly specified")

    def objective(self, x, check_feasibility=False):
        return self.smooth_objective(x,mode='func', check_feasibility=check_feasibility) + self.nonsmooth_objective(x, check_feasibility=check_feasibility)

    def proximal_optimum(self, x, lipschitz=1):
        """
        Returns

        .. math::

           \inf_{v \in \mathbb{R}^p} \frac{L}{2}
           \|x-v\|^2_2 + \lambda h(v)

        where *p*=x.shape[0] and :math:`h(v)` = self.seminorm(v).

        """
        argmin = self.proximal(x, lipschitz)
        return argmin, lipschitz * norm(x-argmin)**2 / 2. + self.nonsmooth_objective(argmin)


    def proximal_step(self, x, grad, lipschitz, prox_control=None):
        """
        Compute the proximal optimization

        prox_control: If not None, then a dictionary of parameters for the prox procedure
        """
        if self.compute_difference:
            z = x - grad / lipschitz
            if prox_control is None:
                return self.proximal(z, lipschitz)
            else:
                return self.proximal(z, lipschitz, prox_control=prox_control)
        else:
            return self.proximal(x, grad, lipschitz)
        
    def get_lipschitz(self):
        return self._lipschitz
    def set_lipschitz(self, value):
        if value < 0:
            raise ValueError('Lipschitz constant must be non-negative')
        self._lipschitz = value
    lipschitz = property(get_lipschitz, set_lipschitz)

class nonsmooth(composite):
    """
    A composite subclass that explicitly 0
    as smooth_objective.
    """

    def __init__(self, nonsmooth, proximal, initial, smooth_multiplier=1, lipschitz=None):
        def _smooth_objective(self, x, mode='both', check_feasibility=False):
            if mode == 'both':
                return 0., zeros(x.shape)
            elif mode == 'func':
                return 0.
            elif mode == 'grad':
                return zeros(x.shape)
            raise ValueError("Mode not specified correctly")

        composite.__init__(self, _smooth_objective, nonsmooth, proximal, initial, smooth_multiplier=smooth_multiplier, lipschitz=lipschitz)


class smooth(composite):

    """
    A composite subclass that has 0 as 
    nonsmooth_objective and the proximal
    is a null-op.
    """

    def __init__(self, smooth_objective, initial, smooth_multiplier=1, lipschitz=None):
        def zero_func(x, check_feasibility=False):
            return 0
        def zero_proximal(x, lipschitz):
            return x
        composite.__init__(self, smooth_objective, zero_func, zero_proximal,
                           initial,
                           smooth_multiplier=smooth_multiplier,
                           lipschitz=lipschitz)


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
        self.epsilon = epsilon
        if self.epsilon <= 0:
            raise ValueError('to smooth, epsilon must be positive')
        self.primal_shape = atom.primal_shape

        self.atom = atom
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
            argmin, optimal_value = dual_atom.proximal_optimum(ueps, self.epsilon)                    
            objective = self.epsilon / 2. * norm(ueps)**2 - optimal_value + constant_term
            grad = linear_transform.adjoint_map(argmin)
            if self.store_argmin:
                self.argmin = argmin
            return objective, grad
        elif mode == 'grad':
            argmin = dual_atom.proximal(ueps, self.epsilon)     
            grad = linear_transform.adjoint_map(argmin)
            if self.store_argmin:
                self.argmin = argmin
            return grad 
        elif mode == 'func':
            _, optimal_value = dual_atom.proximal_optimum(ueps, self.epsilon)                    
            objective = self.epsilon / 2. * norm(ueps)**2 - optimal_value + constant_term
            return objective
        else:
            raise ValueError("mode incorrectly specified")

    def nonsmooth_objective(self, x, check_feasibilty=False):
        return 0

    def proximal(self, x, lipschitz=1):
        return x

    def _get_atom(self):
        return self._atom
    def _set_atom(self, atom):
        self._atom = atom
        self.dual= self._atom.dual
    atom = property(_get_atom, _set_atom)
