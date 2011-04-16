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
        """
        Create a new problem object using the seminorm
        """
        if initial is None:
            return seminorm.problem(self.smooth_eval, 
                                    smooth_multiplier=smooth_multiplier,
                                    initial=self.coefs)
        else:
            return seminorm.problem(self.smooth_eval, 
                                    smooth_multiplier=smooth_multiplier,
                                    initial=initial)

    def proximal(self, x, g, L):
        """
        Take a gradient step
        """
        return x - g / L

    def obj_rough(self, x):
        """
        There is no nonsmooth objective component - return 0
        """
        return 0.
    
class squaredloss(smooth_function):

    """
    A class for combining squared error loss with a general seminorm
    """

    def __init__(self, X, Y, initial=None):
        """
        Generate initial tuple of arguments for update.
        """
        self.X = X
        self.Y = Y
        self.n, self.p = self.X.shape
        if initial is not None:
            self.coefs = initial.copy()
        else:
            self.coefs = np.zeros(self.p)

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

    def __init__(self, Y, initial=None):
        """
        Generate initial tuple of arguments for update.
        """
        self.Y = Y
        self.n = self.p = self.Y.shape[0]
        if initial is not None:
            self.coefs = initial.copy()
        else:
            self.coefs = np.zeros(self.p)

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


    
class logistic_loglikelihood(smooth_function):

    """
    A class for combining the logistic log-likelihood with a general seminorm
    """

    def __init__(self, X, Y, initial=None):
        """
        Generate initial tuple of arguments for update.
        """
        self.X = X
        self.Y = Y
        self.n, self.p = self.X.shape
        if initial is not None:
            self.coefs = initial.copy()
        else:
            self.coefs = np.zeros(self.p)

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
        exp_yhat = np.exp(yhat)
        if mode == 'both':
            ratio = exp_yhat/(1.+exp_yhat)
            return -2*np.dot(self.Y,yhat) + 2* np.sum(np.log(1+exp_yhat)), -2*self._dotT(yhat-ratio)
        elif mode == 'grad':
            ratio = exp_yhat/(1.+exp_yhat)
            return -2*self._dotT(yhat-ratio)
        elif mode == 'func':
            return -2* np.dot(self.Y,yhat) +2* np.sum(np.log(1+exp_yhat))
        else:
            raise ValueError("mode incorrectly specified")

    def set_Y(self, Y):
        self._Y = Y
    def get_Y(self):
        return self._Y
    Y = property(get_Y, set_Y)
