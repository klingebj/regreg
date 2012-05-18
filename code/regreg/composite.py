from numpy.linalg import norm
from numpy import zeros, array

# local import

from identity_quadratic import identity_quadratic as sq

class composite(object):
    """
    A generic way to specify a problem in composite form.
    """

    def __init__(self, primal_shape, offset=None,
                 quadratic=None, initial=None):

        self.offset = offset
        if offset is not None:
            self.offset = array(offset)

        if type(primal_shape) == type(1):
            self.primal_shape = (primal_shape,)
        else:
            self.primal_shape = primal_shape
        self.dual_shape = self.primal_shape

        if quadratic is not None:
            self.quadratic = quadratic
        else:
            self.quadratic = sq(0,0,0,0)

        if initial is None:
            self.coefs = zeros(self.primal_shape)
        else:
            self.coefs = initial.copy()

    def nonsmooth_objective(self, x, check_feasibility=False):
        if self.quadratic is not None:
            return self.quadratic.objective(x, 'func')
        return 0

    def smooth_objective(self, x, mode='both', check_feasibility=False):
        '''
        The smooth_objective and the quadratic_objective combined.
        '''
        raise NotImplementedError

    def objective(self, x, check_feasibility=False):
        return self.smooth_objective(x,mode='func', check_feasibility=check_feasibility) + self.nonsmooth_objective(x, check_feasibility=check_feasibility)

    def proximal_optimum(self, quadratic):
        """
        Returns

        .. math::

           \inf_{v \in \mathbb{R}^p} \frac{L}{2}
           \|x-v\|^2_2 + \lambda h(v)

        where *p*=x.shape[0] and :math:`h(v)` = self.seminorm(v).

        Here, h represents the nonsmooth part and the quadratic
        part of the composite object.

        """
        argmin = self.proximal(quadratic)
        if self.quadratic is None:
            return argmin, lipschitz * norm(x-argmin)**2 / 2. + self.nonsmooth_objective(argmin)  
        else:
            return argmin, lipschitz * norm(x-argmin)**2 / 2. + self.nonsmooth_objective(argmin) + self.quadratic.objective(argmin, 'func') 

    def proximal_step(self, quadratic, prox_control=None):
        """
        Compute the proximal optimization

        prox_control: If not None, then a dictionary of parameters for the prox procedure
        """
        # This seems like a null op -- if all proximals accept optional prox_control
        if prox_control is None:
            return self.proximal(quadratic)
        else:
            return self.proximal(quadratic, prox_control=prox_control)

    def apply_offset(self, x):
        if self.offset is not None:
            return x + self.offset
        return x

        
    def set_quadratic(self, quadratic):
        self._quadratic = quadratic

    def get_quadratic(self):
        if not hasattr(self, "_quadratic"):
            self._quadratic = sq(None, None, None, None)
        return self._quadratic
    quadratic = property(get_quadratic, set_quadratic)

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

    def proximal(self, quadratic):
        if self.quadratic is None:
            totalq = quadratic
        else:
            totalq = self.quadratic + quadratic
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
        q = sq(self.epsilon, ueps, 0, 0)
        if mode == 'both':
            argmin, optimal_value = dual_atom.proximal_optimum(q)
            objective = self.epsilon / 2. * norm(ueps)**2 - optimal_value + constant_term
            grad = linear_transform.adjoint_map(argmin)
            if self.store_argmin:
                self.argmin = argmin
            return objective, grad
        elif mode == 'grad':
            argmin = dual_atom.proximal(q)
            grad = linear_transform.adjoint_map(argmin)
            if self.store_argmin:
                self.argmin = argmin
            return grad 
        elif mode == 'func':
            _, optimal_value = dual_atom.proximal_optimum(q)
            objective = self.epsilon / 2. * norm(ueps)**2 - optimal_value + constant_term
            return objective
        else:
            raise ValueError("mode incorrectly specified")


