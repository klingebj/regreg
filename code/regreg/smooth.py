import numpy as np
from scipy import sparse
import warnings

class smooth_function(object):
    """
    A container class for smooth_atom classes
    """

    def __init__(self, *atoms, **keywords):
        if not set(keywords.keys()).issubset(['l']):
            warnings.warn('only keyword argument should be multiplier, "l", got %s' % `keywords`)
        self.l = 1
        if keywords.has_key('l'):
            self.l *= keywords['l']
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
                return self.scale(v)
            else:
                return self.scale(self.atoms[0].smooth_eval(beta, mode=mode))
        elif mode == 'grad':
            if self.M > 1:
                g = np.zeros(self.p)
                for atom in self.atoms:
                    g += atom.smooth_eval(beta, mode=mode)
                return self.scale(g)
            else:
                return self.scale(self.atoms[0].smooth_eval(beta, mode=mode))
        elif mode == 'both':
            if self.M > 1:
                v = 0.
                g = np.zeros(self.p)
                for atom in self.atoms:
                    output = atom.smooth_eval(beta, mode=mode)
                    v += output[0]
                    g += output[1]
                return self.scale(v), self.scale(g)
            else:
                v, g = self.atoms[0].smooth_eval(beta, mode=mode)
                return self.scale(v), self.scale(g)
        else:
            raise ValueError("Mode specified incorrectly")

    def scale(self, arg):
        if self.l is not None:
            return self.l * arg
        return arg

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
        # XXX the class now has self.l -- no need form smooth_multiplier
        if initial is None:
            return seminorm.problem(self.smooth_eval, 
                                    smooth_multiplier=smooth_multiplier,
                                    initial=self.coefs)
        else:
            return seminorm.problem(self.smooth_eval, 
                                    smooth_multiplier=smooth_multiplier,
                                    initial=initial)

class affine_atom(smooth_function):

    def __init__(self, sm_atom, X=None, Y=None, l=1):
        self.X = X
        self._Y = Y
        self.sm_atom = sm_atom
        smooth_function.__init__(self, self, l=l)

    def smooth_eval(self, beta, mode='both'):
        eta = self._dot(beta)
        if self.Y is not None:
            eta += self.Y
        if mode == 'both':
            v, g = self.sm_atom.smooth_eval(eta, mode='both')
            g = self._dotT(g)
            return self.scale(v), self.scale(g)
        elif mode == 'grad':
            g = self.sm_atom.smooth_eval(eta, mode='grad')
            g = self._dotT(g)
            return self.scale(g)
        elif mode == 'func':
            v = self.sm_atom.smooth_eval(eta, mode='func')
            return self.scale(v)

    def _dot(self, beta):
        if self.X is None:
            return beta
        elif not sparse.isspmatrix(self.X):
            return np.dot(self.X,beta)
        else:
            return self.X * beta

    def _dotT(self, r):
        if self.X is None:
            return r
        if not sparse.isspmatrix(self.X):
            return np.dot(self.X.T, r)
        else:
            return self.X.T * r

    def set_Y(self, Y):
        self._Y = Y
    def get_Y(self):
        return self._Y
    Y = property(get_Y, set_Y)


class smooth_atom(smooth_function):

    """
    A class for representing a smooth function and its gradient
    """

    def __init__(self, p, l=1):
        self.p = p
        self.l = l
        self.coefs = np.zeros(self.p)
        raise NotImplementedError

    def smooth_eval(self):
        raise NotImplementedError
    
    @classmethod
    def affine(cls, X, Y, l=1):
        smoothf = cls(X.shape[1], l=l)
        return affine_atom(smoothf, X, Y)


def squaredloss(X, Y, l=1):
    # the affine method gets rid of the need for the squaredloss class
    # as previously written squared loss had a factor of 2
    return l2normsq.affine(-X, Y, l=l/2.)

class l2normsq(smooth_atom):
    """
    The square of the l2 norm
    """

    #TODO: generalize input to allow for a matrix D, making a generalized l2 norm with syntax like l2norm seminorm_atom

    def __init__(self, p, l=None):
        self.p = p
        self.l = l

    def smooth_eval(self, beta, mode='both'):
        """
        Evaluate a smooth function and/or its gradient

        if mode == 'both', return both function value and gradient
        if mode == 'grad', return only the gradient
        if mode == 'func', return only the function value
        """

        if mode == 'both':
            return self.scale(np.linalg.norm(beta)**2), self.scale(2 * beta)
        elif mode == 'grad':
            return self.scale(2 * beta)
        elif mode == 'func':
            return self.scale(np.linalg.norm(beta)**2)
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
        self.l = l


    def smooth_eval(self, beta, mode='both'):
        """
        Evaluate a smooth function and/or its gradient

        if mode == 'both', return both function value and gradient
        if mode == 'grad', return only the gradient
        if mode == 'func', return only the function value
        """

        if mode == 'both':
            return self.scale(((self.Y - beta)**2).sum() / 2.), self.scale(beta - self.Y)
        elif mode == 'grad':
            return self.scale(beta - self.Y)
        elif mode == 'func':
            return self.scale(((self.Y - beta)**2).sum() / 2.)
        else:
            raise ValueError("mode incorrectly specified")

    def set_Y(self, Y):
        self._Y = Y
    def get_Y(self):
        return self._Y
    Y = property(get_Y, set_Y)

    
class logistic_loglikelihood(affine_atom):

    """
    A class for combining the logistic log-likelihood with a general seminorm
    """

    def __init__(self, X, Y, l=1):
        affine_atom.__init__(self, None, X, Y, l=l)

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
            return -2 * self.scale((np.dot(self.Y,yhat) - np.sum(np.log(1+exp_yhat)))), -2 * self.scale(self._dotT(self.Y-ratio))
        elif mode == 'grad':
            ratio = exp_yhat/(1.+exp_yhat)
            return - 2 * self.scale(self._dotT(self.Y-ratio))
        elif mode == 'func':
            return -2 * self.scale(np.dot(self.Y,yhat) - np.sum(np.log(1+exp_yhat)))
        else:
            raise ValueError("mode incorrectly specified")


class smoothed_seminorm(smooth_function):

    def __init__(self, semi, epsilon=0.01):


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
        self.seminorm = semi
        self.epsilon = epsilon
        if epsilon <= 0:
            raise ValueError('to smooth, epsilon must be positive')
        self.p = semi.primal_dim
        
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

