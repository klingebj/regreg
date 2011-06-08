from numpy.linalg import norm

class composite(object):
    """
    A generic way to specify a problem in composite form.
    """

    def __init__(self, smooth_objective, nonsmooth_objective, proximal, initial, smooth_multiplier=1):
        # Do we need to store this?
        #self.initial = initial.copy()
        self.coefs = initial.copy()
        self.nonsmooth_objective = nonsmooth_objective
        self._smooth_objective = smooth_objective
        self.proximal = proximal
        self.smooth_multiplier = smooth_multiplier

    def smooth_objective(self, x, mode='both'):
        output = self._smooth_objective(x, mode=mode)
        if mode == 'both':
            return self.smooth_multiplier * output[0], self.smooth_multiplier * output[1]
        elif mode == 'grad' or mode == 'func':
            return self.smooth_multiplier * output
        else:
            raise ValueError("Mode incorrectly specified")

    def objective(self, x):
        return self.smooth_objective(x,mode='func') + self.nonsmooth_objective(x)

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


    def proximal_step(self, x, grad, lipshitz, prox_control=None):
        """
        Compute the proximal optimization

        prox_control: If not None, then a dictionary of parameters for the prox procedure
        """
        z = x - grad / lipshitz
        
        if prox_control is None:
            return self.proximal(z, lipshitz)
        else:
            return self.proximal(z, lipshitz, **prox_control)

