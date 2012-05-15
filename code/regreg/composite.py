from numpy.linalg import norm
from numpy import zeros

class composite(object):
    """
    A generic way to specify a problem in composite form.
    """

    def __init__(self, smooth_objective, nonsmooth_objective, proximal, initial, smooth_multiplier=1, lipschitz=None, compute_difference=True,
                 quadratic_spec=(None,None,None)):

        self.coefs = initial.copy()
        self._nonsmooth_objective = nonsmooth_objective
        self._smooth_objective = smooth_objective
        self.proximal = proximal
        self.smooth_multiplier = smooth_multiplier
        self._lipschitz = lipschitz

        # this is used for atoms that have a different
        # signature for the proximal method
        self.compute_difference = compute_difference

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

    def objective(self, x, check_feasibility=False):
        return self.smooth_objective(x,mode='func', check_feasibility=check_feasibility) + self.nonsmooth_objective(x, check_feasibility=check_feasibility)

    def proximal_optimum(self, x, lipschitz=1):
        """
        Returns

        .. math::

           \inf_{v \in \mathbb{R}^p} \frac{L}{2}
           \|x-v\|^2_2 + \lambda h(v)

        where *p*=x.shape[0] and :math:`h(v)` = self.seminorm(v).

        Here, h represents the nonsmooth part and the quadratic
        part of the composite object.

        """
        argmin = self.proximal(x, lipschitz)
        if self.quadratic is None:
            return argmin, lipschitz * norm(x-argmin)**2 / 2. + self.nonsmooth_objective(argmin)  
        else:
            return argmin, lipschitz * norm(x-argmin)**2 / 2. + self.nonsmooth_objective(argmin) + self.quadratic.objective(argmin, 'func') 

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
        
    def set_quadratic_spec(self, quadratic_spec):
        self._quadratic = identity_quadratic(*quadratic_spec)

    def get_quadratic_spec(self):
        if hasattr(self, "_quadratic") and self._quadratic.anything_to_return:
            return self._quadratic
        else:
            return None
    quadratic = property(get_quadratic_spec, set_quadratic_spec)

    def get_lipschitz(self):
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


#    def __init__(self, nonsmooth, proximal, initial, smooth_multiplier=1, lipschitz=None, quadratic_spec=(None, None)):
#         def _smooth_objective(self, x, mode='both', check_feasibility=False):
#             if mode == 'both':
#                 return 0., zeros(x.shape)
#             elif mode == 'func':
#                 return 0.
#             elif mode == 'grad':
#                 return zeros(x.shape)
#             raise ValueError("Mode not specified correctly")

#        composite.__init__(self, nonsmooth._smooth_objective, nonsmooth, proximal, initial, smooth_multiplier=smooth_multiplier, lipschitz=lipschitz)


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

    def proximal(self, x, lipschitz=1):
        if self.quadratic is None:
            return x
        else:
            q = self.quadratic
            total_q = lipschitz + q.coef
            total_linear = lipschitz * x
            if q.offset is not None:
                total_linear -= q.coef * q.offset
            if q.linear is not None:
                total_linear -= q.linear
            return total_linear / total_q    

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
            argmin, optimal_value = dual_atom.proximal_optimum(ueps, self.epsilon)                    
            print optimal_value
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


class identity_quadratic(object):

    def __init__(self, coef, offset, linear):
        self.coef = coef
        self.offset = offset
        self.linear = linear
        if self.coef is not None or self.linear is not None:
            self.anything_to_return = True
        else:
            self.anything_to_return = False

    def objective(self, x, mode='both'):
        coef, offset, linear = self.coef, self.offset, self.linear
        if linear is None:
            linear = 0
        if offset is not None:
            r = x + offset
        else:
            r = x
        if mode == 'both':
            if linear is not None:
                return (norm(r)**2 * coef / 2. + (linear * x).sum(),
                    coef * r + linear)
            else:
                return (norm(r)**2 * coef / 2.,
                    coef * r)
        elif mode == 'func':
            if linear is not None:
                return norm(r)**2 * coef / 2. + (linear * x).sum()
            else:
                return norm(r)**2 * coef / 2.
        elif mode == 'grad':
            if linear is not None:
                return coef * r + linear
            else:
                return coef * r
        else:
            raise ValueError("Mode incorrectly specified")
                        
    def update_smooth_output(self, x, smooth_output, mode='both'):
        quadratic_output = self.objective(x, mode=mode)
        if mode == 'both':
            smooth_output = (smooth_output[0] + quadratic_output[0],
                             smooth_output[1] + quadratic_output[1])
        else:
            smooth_output = smooth_output + quadratic_output

        return smooth_output

    def __repr__(self):
        return 'identity_quadratic(%f, %s, %s)' % (self.coef, `self.offset`, `self.linear`)
