import numpy as np
from scipy import sparse


class smooth_function(object):
    """
    A container class for smooth_atom classes
    """

    def __init__(self, *atoms):
        self.atoms = []
        self.p = None
        for atom in atoms:
            if self.p is None:
                self.p = atom.p
            elif atom.p != self.p:
                raise ValueError("Smooth function dimensions don't agree")
            self.atoms.append(atom)

        self.coefs = np.zeros(self.p)

        self.M = len(atoms)

    #TODO: add addition overload

    def smooth_eval(self, beta, mode='both'):
        """
        Evaluate a smooth function and/or its gradient

        if mode == 'both', return both function value and gradient
        if mode == 'grad', return only the gradient
        if mode == 'func', return only the function value
        """

        if mode == 'func':
            if self.M > 1:
                v = 0.
                for atom in self.atoms:
                    v += atom.smooth_eval(beta, mode=mode)
                return v
            else:
                return self.atoms[0].smooth_eval(beta, mode=mode)
        elif mode == 'grad':
            if self.M > 1:
                g = np.zeros(self.p)
                for atom in self.atoms:
                    g += atom.smooth_eval(beta, mode=mode)
                return g
            else:
                return self.atoms[0].smooth_eval(beta, mode=mode)
        elif mode == 'both':
            if self.M > 1:
                v = 0.
                g = np.zeros(self.p)
                for atom in self.atoms:
                    output = atom.smooth_eval(beta, mode=mode)
                    v += output[0]
                    g += output[1]
                return v, g
            else:
                return self.atoms[0].smooth_eval(beta, mode=mode)
        else:
            raise ValueError("Mode specified incorrectly")

    
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

class smooth_atom(object):

    """
    A class for representing a smooth function and its gradient
    """

    l = 1.

    def __init__(self):
        raise NotImplementedError

    def smooth_eval(self):
        raise NotImplementedError
    
class squaredloss(smooth_atom):

    """
    A class for combining squared error loss with a general seminorm
    """

    def __init__(self, X, Y, l = None):
        """
        Generate initial tuple of arguments for update.
        """
        self.X = X
        self.Y = Y
        self.n, self.p = self.X.shape

        if l is not None:
            self.l = l

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
            return self.l * ((self.Y - yhat)**2).sum() / 2. , self.l * self._dotT(yhat-self.Y)
        elif mode == 'grad':
            return self.l * self._dotT(yhat-self.Y)
        elif mode == 'func':
            return self.l * ((self.Y - yhat)**2).sum() / 2.
        else:
            raise ValueError("mode incorrectly specified")
        

    def set_Y(self, Y):
        self._Y = Y
    def get_Y(self):
        return self._Y
    Y = property(get_Y, set_Y)


class l2normsq(smooth_atom):
    """
    The square of the l2 norm
    """

    #TODO: generalize input to allow for a matrix D, making a generalized l2 norm with syntax like l2norm seminorm_atom

    def __init__(self, p, l=None):
        self.p = p

        if l is not None:
            self.l = l

    def smooth_eval(self, beta, mode='both'):
        """
        Evaluate a smooth function and/or its gradient

        if mode == 'both', return both function value and gradient
        if mode == 'grad', return only the gradient
        if mode == 'func', return only the function value
        """

        if mode == 'both':
            return self.l * np.linalg.norm(beta)**2, self.l * 2 * beta
        elif mode == 'grad':
            return self.l * 2 * beta
        elif mode == 'func':
            return self.l * np.linalg.norm(beta)**2
        else:
            raise ValueError("mode incorrectly specified")


            
class signal_approximator(smooth_atom):

    """
    A class for combining squared error loss with a general seminorm
    """

    def __init__(self, Y, l=None):
        """
        Generate initial tuple of arguments for update.
        """
        self.Y = Y
        self.n = self.p = self.Y.shape[0]

        if l is not None:
            self.l = l


    def smooth_eval(self, beta, mode='both'):
        """
        Evaluate a smooth function and/or its gradient

        if mode == 'both', return both function value and gradient
        if mode == 'grad', return only the gradient
        if mode == 'func', return only the function value
        """

        if mode == 'both':
            return self.l * ((self.Y - beta)**2).sum() / 2., self.l * (beta - self.Y)
        elif mode == 'grad':
            return self.l * (beta - self.Y)
        elif mode == 'func':
            return self.l * ((self.Y - beta)**2).sum() / 2.
        else:
            raise ValueError("mode incorrectly specified")

    def set_Y(self, Y):
        self._Y = Y
    def get_Y(self):
        return self._Y
    Y = property(get_Y, set_Y)


    
class logistic_loglikelihood(smooth_atom):

    """
    A class for combining the logistic log-likelihood with a general seminorm
    """

    def __init__(self, X, Y, l = None):
        """
        Generate initial tuple of arguments for update.
        """
        self.X = X
        self.Y = Y
        self.n, self.p = self.X.shape

        if l is not None:
            self.l = l

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
            return -2 * self.l * (np.dot(self.Y,yhat) - np.sum(np.log(1+exp_yhat))), -2 * self.l * self._dotT(self.Y-ratio)
        elif mode == 'grad':
            ratio = exp_yhat/(1.+exp_yhat)
            return - 2 * self.l * self._dotT(self.Y-ratio)
        elif mode == 'func':
            return -2 * self.l * (np.dot(self.Y,yhat) - np.sum(np.log(1+exp_yhat)))
        else:
            raise ValueError("mode incorrectly specified")

    def set_Y(self, Y):
        self._Y = Y
    def get_Y(self):
        return self._Y
    Y = property(get_Y, set_Y)



class huber_loss(squaredloss):

    """
    A class for representing the Huber loss function
    """

    def __init__(self, X, Y, delta = 1., l = None):
        """
        Generate initial tuple of arguments for update.
        """
        self.X = X
        self.Y = Y
        self.delta = delta
        self.n, self.p = self.X.shape

        if l is not None:
            self.l = l
    
    def smooth_eval(self, beta, mode='both'):
        """
        Evaluate a smooth function and/or its gradient

        if mode == 'both', return both function value and gradient
        if mode == 'grad', return only the gradient
        if mode == 'func', return only the function value
        """
        
        resid = self.Y - self._dot(beta)
        quad = np.fabs(resid) <= self.delta

        if mode == 'both':
            #This could be made more efficient in cython...
            func = 0.5 * np.sum(quad*(resid**2) + (1-quad)*(2*self.delta*resid - self.delta**2))
            grad = - 0.5 * self._dotT( quad*resid + (1-quad)*2*self.delta )
            return self.l * func, self.l * grad
        elif mode == 'grad':
            grad = -0.5 * self._dotT( quad*resid + (1-quad)*2*self.delta )
            return self.l * grad
        elif mode == 'func':
            func = 0.5 * np.sum(quad*(resid**2) + (1-quad)*(2*self.delta*resid - self.delta**2))
            return self.l * func
        else:
            raise ValueError("mode incorrectly specified")
