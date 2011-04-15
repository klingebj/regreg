import numpy as np
from scipy import sparse

class smooth_function(object):

    """
    A class for representing a smooth function and its gradient
    """

    def __init__(self):
        raise NotImplementedError

    def smooth_eval(self):
        raise NotImplementedError

    def add_seminorm(self, seminorm, initial=None, smooth_multiplier=1):
        return seminorm.problem(self.smooth_eval, 
                                smooth_multiplier=smooth_multiplier,
                                initial=initial)


    
class squaredloss(smooth_function):

    """
    A class for combining squared error loss with a general seminorm
    """

    def __init__(self, X, Y):
        """
        Generate initial tuple of arguments for update.
        """
        self.X = X
        self.Y = Y
        self.n, self.p = self.X.shape

    def _dot(self, beta):
        if not sparse.isspmatrix(self.X):
            return np.dot(self.X,beta)
        else:
            return self.X * beta

    def _dotT(self, r):
        if not sparse.isspmatrix(self.X):
            return np.dot(self.X.T, r)
        else:
            return self.X.T * r

    def smooth_eval(self, beta, mode='both'):
        """
        Evaluate a smooth function and/or its gradient

        if mode == 'both', return both function value and gradient
        if mode == 'grad', return only the gradient
        if mode == 'func', return only the function value
        """
        yhat = self._dot(beta)
        if mode == 'both':
            return ((self.Y - yhat)**2).sum() / 2. , self._dotT(yhat-self.Y)
        elif mode == 'grad':
            return self._dotT(yhat-self.Y)
        elif mode == 'func':
            return ((self.Y - yhat)**2).sum() / 2.
        else:
            raise ValueError("mode incorrectly specified")

    def set_Y(self, Y):
        self._Y = Y
    def get_Y(self):
        return self._Y
    Y = property(get_Y, set_Y)

class signal_approximator(smooth_function):

    """
    A class for combining squared error loss with a general seminorm
    """

    def __init__(self, Y):
        """
        Generate initial tuple of arguments for update.
        """
        self.Y = Y
        self.n = self.p = self.Y.shape[0]

    def smooth_eval(self, beta, mode='both'):
        """
        Evaluate a smooth function and/or its gradient

        if mode == 'both', return both function value and gradient
        if mode == 'grad', return only the gradient
        if mode == 'func', return only the function value
        """

        if mode == 'both':
            return ((self.Y - beta)**2).sum() / 2., beta - self.Y
        elif mode == 'grad':
            return beta - self.Y
        elif mode == 'func':
            return ((self.Y - beta)**2).sum() / 2.
        else:
            raise ValueError("mode incorrectly specified")

    def set_Y(self, Y):
        self._Y = Y
    def get_Y(self):
        return self._Y
    Y = property(get_Y, set_Y)
