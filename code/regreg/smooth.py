
import numpy as np
from scipy import sparse
from affine import affine_transform
import warnings

class smooth_function(object):
    """
    A container class for smooth_atom classes
    """

    # TODO? use a list for atoms instead of *atoms?
    def __init__(self, *atoms, **keywords):
        if not set(keywords.keys()).issubset(['lagrange']):
            warnings.warn('only keyword argument should be multiplier, "l", got %s' % `keywords`)
        self.lagrange =1
        if keywords.has_key('lagrange'):
            self.lagrange *= keywords['lagrange']
        self.atoms = []
        self.primal_shape = -1
        for atom in atoms:
            if self.primal_shape == -1:
                self.primal_shape = atom.primal_shape
            else:
                if atom.primal_shape != self.primal_shape:
                    raise ValueError("primal dimensions don't agree")
            self.atoms.append(atom)
        self.coefs = np.zeros(self.primal_shape)

    #TODO: add addition overload

    def nonsmooth_objective(self, x):
        return 0

    def proximal(self, x, lipschitz=1):
        return x

    def smooth_objective(self, x, mode='both'):
        """
        Evaluate a smooth function and/or its gradient

        if mode == 'both', return both function value and gradient
        if mode == 'grad', return only the gradient
        if mode == 'func', return only the function value
        """

        if mode == 'func':
            if len(self.atoms) > 1:
                v = 0.
                for atom in self.atoms:
                    v += atom.smooth_objective(x, mode=mode)
                return self.scale(v)
            else:
                return self.scale(self.atoms[0].smooth_objective(x, mode=mode))
        elif mode == 'grad':
            if len(self.atoms) > 1:
                g = np.zeros(self.primal_shape)
                for atom in self.atoms:
                    g += atom.smooth_objective(x, mode=mode)
                return self.scale(g)
            else:
                return self.scale(self.atoms[0].smooth_objective(x, mode=mode))
        elif mode == 'both':
            if len(self.atoms) > 1:
                v = 0.
                g = np.zeros(self.primal_shape)
                for atom in self.atoms:
                    output = atom.smooth_objective(x, mode=mode)
                    v += output[0]
                    g += output[1]
                return self.scale(v), self.scale(g)
            else:
                v, g = self.atoms[0].smooth_objective(x, mode=mode)
                return self.scale(v), self.scale(g)
        else:
            raise ValueError("Mode specified incorrectly")

    def scale(self, obj, copy=False):
        if self.lagrange != 1:
            return obj * self.lagrange
        if copy:
            return obj.copy()
        return obj
    
    def proximal(self, x, g, lipshitz):
        """
        Take a gradient step
        """
        return x - g / lipshitz

    def obj_rough(self, x):
        """
        There is no nonsmooth objective component - return 0
        """
        return 0.


class affine_smooth(smooth_function):

    # if smooth_obj is a class, an object is created
    # smooth_obj(*args, **keywords)
    # else, it is assumed to be an instance of smooth_function
 
    def __init__(self, smooth_obj, linear_operator=None, offset=None, diag=False, lagrange=1, args=(), keywords={}):
        self.affine_transform = affine_transform(linear_operator, offset, diag)
        self.primal_shape = self.affine_transform.primal_shape
        self.coefs = np.zeros(self.primal_shape)
        keywords = keywords.copy(); keywords['lagrange'] = lagrange
        if type(smooth_obj) == type(type): # a class object
            smooth_class = smooth_obj
            self.sm_atom = smooth_class(self.primal_shape, *args, **keywords)
        else:
            self.sm_atom = smooth_obj
        self.atoms = [self]

    def _getl(self):
        return self.sm_atom.lagrange

    def _setl(self, l):
        self.sm_atom.lagrange = lagrange
    l = property(_getl, _setl)

    def smooth_objective(self, x, mode='both'):
        eta = self.affine_transform.affine_map(x)
        if mode == 'both':
            v, g = self.sm_atom.smooth_objective(eta, mode='both')
            g = self.affine_transform.adjoint_map(g)
            return v, g
        elif mode == 'grad':
            g = self.sm_atom.smooth_objective(eta, mode='grad')
            g = self.affine_transform.adjoint_map(g)
            return g 
        elif mode == 'func':
            v = self.sm_atom.smooth_objective(eta, mode='func')
            return v 


class smooth_atom(smooth_function):

    """
    A class for representing a smooth function and its gradient
    """

    def __init__(self, primal_shape, lagrange=1):
        self.primal_shape = primal_shape
        self.lagrange = lagrange
        self.coefs = np.zeros(self.primal_shape)
        raise NotImplementedError

    def smooth_objective(self):
        raise NotImplementedError
    
    @classmethod
    def affine(cls, linear_operator, offset, lagrange=1, diag=False,
               args=(), keywords={}):
        """
        Args and keywords passed to cls constructor along with
        l and primal_shape
        """
        return affine_smooth(cls, linear_operator, offset, diag=diag, lagrange=lagrange,
                             args=args, keywords=keywords)

    @classmethod
    def linear(cls, linear_operator, lagrange=1, diag=False,
               args=(), keywords={}):
        """
        Args and keywords passed to cls constructor along with
        l and primal_shape
        """
        return affine_smooth(cls, linear_operator, None, diag=diag, lagrange=lagrange,
                             args=args, keywords=keywords)

    @classmethod
    def shift(cls, offset, lagrange=1, diag=False,
               args=(), keywords={}):
        """
        Args and keywords passed to cls constructor along with
        l and primal_shape
        """
        return affine_smooth(cls, None, offset, diag=diag, lagrange=lagrange,
                             args=args, keywords=keywords)


def squaredloss(linear_operator, offset, lagrange=1):
    # the affine method gets rid of the need for the squaredloss class
    # as previously written squared loss had a factor of 2

    #return l2normsq.affine(-linear_operator, offset, lagrange=lagrange/2., initial=np.zeros(linear_operator.shape[1]))
    return l2normsq.affine(-linear_operator, offset, lagrange=lagrange/2.)

class l2normsq(smooth_atom):
    """
    The square of the l2 norm
    """

    def __init__(self, primal_shape, lagrange=None, Q=None, Qdiag=False):
        self.Q = Q
        if self.Q is not None:
            self.Q_transform = affine_transform(Q, 0, Qdiag)
        if type(primal_shape) == type(1):
            self.primal_shape = (primal_shape,)
        else:
            self.primal_shape = primal_shape
        self.lagrange = lagrange

    def smooth_objective(self, x, mode='both'):
        """
        Evaluate a smooth function and/or its gradient

        if mode == 'both', return both function value and gradient
        if mode == 'grad', return only the gradient
        if mode == 'func', return only the function value
        """

        if self.Q is None:
            if mode == 'both':
                return self.scale(np.linalg.norm(x)**2), self.scale(2 * x)
            elif mode == 'grad':
                return self.scale(2 * x)
            elif mode == 'func':
                return self.scale(np.linalg.norm(x)**2)
            else:
                raise ValueError("mode incorrectly specified")
        else:
            if mode == 'both':
                return self.scale(np.sum(x * self.Q_transform.linear_map(x))), self.scale(2 * self.Q_transform.linear_map(x))
            elif mode == 'grad':
                return self.scale(2 * self.Q_transform.linear_map(x))
            elif mode == 'func':
                return self.scale(np.sum(x * self.Q_transform.linear_map(x)))
            else:
                raise ValueError("mode incorrectly specified")
            
class linear(smooth_atom):

    def __init__(self, vector, lagrange=1):
        self.vector = lagrange * vector
        self.primal_shape = vector.shape
        self.lagrange =1
        self.coefs = np.zeros(self.primal_shape)

    def smooth_objective(self, x, mode='both'):
        """
        Evaluate a smooth function and/or its gradient

        if mode == 'both', return both function value and gradient
        if mode == 'grad', return only the gradient
        if mode == 'func', return only the function value
        """

        if mode == 'both':
            return np.dot(self.vector, x), self.vector
        elif mode == 'grad':
            return self.vector
        elif mode == 'func':
            return np.dot(self.vector, x)
        else:
            raise ValueError("mode incorrectly specified")
    

def signal_approximator(offset, lagrange=1):
    return l2normsq.shift(-offset, lagrange)

class logistic_loglikelihood(smooth_atom):

    """
    A class for combining the logistic log-likelihood with a general seminorm
    """

    def __init__(self, linear_operator, binary_response, offset=None, lagrange=1):
        self.affine_transform = affine_transform(linear_operator, offset)
        self.binary_response = binary_response
        self.primal_shape = self.affine_transform.primal_shape
        self.lagrange = lagrange

    def smooth_objective(self, x, mode='both'):
        """
        Evaluate a smooth function and/or its gradient

        if mode == 'both', return both function value and gradient
        if mode == 'grad', return only the gradient
        if mode == 'func', return only the function value
        """
        
        yhat = self.affine_transform.linear_map(x)
        exp_yhat = np.exp(yhat)
        if mode == 'both':
            ratio = exp_yhat/(1.+exp_yhat)
            return -2 * self.scale((np.dot(self.binary_response,yhat) - np.sum(np.log(1+exp_yhat)))), -2 * self.scale(self.affine_transform.adjoint_map(self.binary_response-ratio))
        elif mode == 'grad':
            ratio = exp_yhat/(1.+exp_yhat)
            return - 2 * self.scale(self.affine_transform.adjoint_map(self.binary_response-ratio))
        elif mode == 'func':
            return -2 * self.scale(np.dot(self.binary_response,yhat) - np.sum(np.log(1+exp_yhat)))
        else:
            raise ValueError("mode incorrectly specified")

class zero(smooth_function):

    def __init__(self, primal_shape):
        self.primal_shape = primal_shape

    def smooth_objective(self, x, mode='both'):
        """
        Evaluate a smooth function and/or its gradient

        if mode == 'both', return both function value and gradient
        if mode == 'grad', return only the gradient
        if mode == 'func', return only the function value
        """
        
        if mode == 'both':
            return 0, np.zeros(x.shape)
        elif mode == 'grad':
            return np.zeros(x.shape)
        elif mode == 'func':
            return 0
        else:
            raise ValueError("mode incorrectly specified")


