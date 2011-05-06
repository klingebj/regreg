
class dummy_problem(object):
    """
    A generic way to specify a problem
    """

    def __init__(self, smooth_eval, nonsmooth, prox, initial, smooth_multiplier=1):
        # Do we need to store this?
        #self.initial = initial.copy()
        self.coefs = initial.copy()
        self.obj_rough = nonsmooth
        self._smooth_eval = smooth_eval
        self._prox = prox
        self.smooth_multiplier = smooth_multiplier

    def smooth_eval(self, x, mode='both'):
        output = self._smooth_eval(x, mode=mode)
        if mode == 'both':
            return self.smooth_multiplier * output[0], self.smooth_multiplier * output[1]
        elif mode == 'grad' or mode == 'func':
            return self.smooth_multiplier * output
        else:
            raise ValueError("Mode incorrectly specified")

    def obj(self, x):
        return self.smooth_eval(x,mode='func') + self.obj_rough(x)

    def proximal(self, x, g, L, prox_control=None):
        """
        Compute the proximal optimization

        prox_control: If not None, then a dictionary of parameters for the prox procedure
        """
        z = x - g / L
        
        if prox_control is None:
            v = self._prox(z, L)
            return v
        else:
            return self._prox(z, L, **prox_control)

