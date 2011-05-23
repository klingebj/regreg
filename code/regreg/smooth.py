import numpy as np
from scipy import sparse
from affine import affine_transform
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

    def smooth_eval(self, beta, mode='both'):
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
                    v += atom.smooth_eval(beta, mode=mode)
                return self.scale(v)
            else:
                return self.scale(self.atoms[0].smooth_eval(beta, mode=mode))
        elif mode == 'grad':
            if len(self.atoms) > 1:
                g = np.zeros(self.primal_shape)
                for atom in self.atoms:
                    g += atom.smooth_eval(beta, mode=mode)
                return self.scale(g)
            else:
                return self.scale(self.atoms[0].smooth_eval(beta, mode=mode))
        elif mode == 'both':
            if len(self.atoms) > 1:
                v = 0.
                g = np.zeros(self.primal_shape)
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

    def scale(self, obj, copy=False):
        if self.l != 1:
            return obj * self.l
        if copy:
            return obj.copy()
        return obj
    
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
            return seminorm.primal_problem(self.smooth_eval, 
                                    smooth_multiplier=smooth_multiplier,
                                    initial=self.coefs)
        else:
            return seminorm.primal_problem(self.smooth_eval, 
                                    smooth_multiplier=smooth_multiplier,
                                    initial=initial)

class affine_atom(smooth_function):

    def __init__(self, smooth_class, linear_operator=None, offset=None, diag=False, l=1, args=(), keywords={}):
        self.affine_transform = affine_transform(linear_operator, offset, diag)
        self.primal_shape = self.affine_transform.primal_shape
        self.coefs = np.zeros(self.primal_shape)
        keywords = keywords.copy(); keywords['l'] = l
        self.sm_atom = smooth_class(self.primal_shape, *args, **keywords)
        self.atoms = [self]

    def _getl(self):
        return self.sm_atom.l

    def _setl(self, l):
        self.sm_atom.l = l
    l = property(_getl, _setl)

    def smooth_eval(self, beta, mode='both'):
        eta = self.affine_transform.affine_map(beta)
        if mode == 'both':
            v, g = self.sm_atom.smooth_eval(eta, mode='both')
            g = self.affine_transform.adjoint_map(g)
            return v, g
        elif mode == 'grad':
            g = self.sm_atom.smooth_eval(eta, mode='grad')
            g = self.affine_transform.adjoint_map(g)
            return g 
        elif mode == 'func':
            v = self.sm_atom.smooth_eval(eta, mode='func')
            return v 

class smooth_atom(smooth_function):

    """
    A class for representing a smooth function and its gradient
    """

    def __init__(self, primal_shape, l=1):
        self.primal_shape = primal_shape
        self.l = l
        self.coefs = np.zeros(self.primal_shape)
        raise NotImplementedError

    def smooth_eval(self):
        raise NotImplementedError
    
    @classmethod
    def affine(cls, linear_operator, offset, l=1, diag=False,
               args=(), keywords={}):
        """
        Args and keywords passed to cls constructor along with
        l and primal_shape
        """
        return affine_atom(cls, linear_operator, offset, diag=diag, l=l,
                           args=args, keywords=keywords)

    @classmethod
    def linear(cls, linear_operator, l=1, diag=False,
               args=(), keywords={}):
        """
        Args and keywords passed to cls constructor along with
        l and primal_shape
        """
        return affine_atom(cls, linear_operator, None, diag=diag, l=l,
                           args=args, keywords=keywords)

    @classmethod
    def shift(cls, offset, l=1, diag=False,
               args=(), keywords={}):
        """
        Args and keywords passed to cls constructor along with
        l and primal_shape
        """
        return affine_atom(cls, None, offset, diag=diag, l=l,
                           args=args, keywords=keywords)


def squaredloss(linear_operator, offset, l=1):
    # the affine method gets rid of the need for the squaredloss class
    # as previously written squared loss had a factor of 2

    #return l2normsq.affine(-linear_operator, offset, l=l/2., initial=np.zeros(linear_operator.shape[1]))
    return l2normsq.affine(-linear_operator, offset, l=l/2.)

class l2normsq(smooth_atom):
    """
    The square of the l2 norm
    """

    #TODO: generalize input to allow for a matrix D, making a generalized l2 norm with syntax like l2norm seminorm_atom

    def __init__(self, primal_shape, l=None):
        if type(primal_shape) == type(1):
            self.primal_shape = (primal_shape,)
        else:
            self.primal_shape = primal_shape
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
            
class linear(smooth_atom):

    def __init__(self, vector, l=1):
        self.vector = l * vector
        self.primal_shape = vector.shape
        self.l = 1
        self.coefs = np.zeros(self.primal_shape)

    def smooth_eval(self, beta, mode='both'):
        """
        Evaluate a smooth function and/or its gradient

        if mode == 'both', return both function value and gradient
        if mode == 'grad', return only the gradient
        if mode == 'func', return only the function value
        """

        if mode == 'both':
            return np.dot(self.vector, beta), self.vector
        elif mode == 'grad':
            return self.vector
        elif mode == 'func':
            return np.dot(self.vector, beta)
        else:
            raise ValueError("mode incorrectly specified")
    

def signal_approximator(offset, l=1):
    return l2normsq.shift(-offset, l)

class logistic_loglikelihood(smooth_atom):

    """
    A class for combining the logistic log-likelihood with a general seminorm
    """

    def __init__(self, linear_operator, binary_response, offset=None, l=1):
        self.affine_transform = affine_transform(linear_operator, offset)
        self.binary_response = binary_response
        self.primal_shape = self.affine_transform.primal_shape
        self.l = l

    def smooth_eval(self, beta, mode='both'):
        """
        Evaluate a smooth function and/or its gradient

        if mode == 'both', return both function value and gradient
        if mode == 'grad', return only the gradient
        if mode == 'func', return only the function value
        """
        
        yhat = self.affine_transform.linear_map(beta)
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


class smoothed_seminorm(smooth_function):

    def __init__(self, *atoms, **keywords):


        """
        Given a seminorm :math:`h_K(D\beta)`, this
        class creates a smoothed version

        .. math::

            h_{K,\varepsilon}(D\beta+\alpha) = \sup_{u \in K}u'(D\beta+\alpha) - \frac{\epsilon}{2}
            \|u\|^2_2

        The objective value is given by

        .. math::

           h_{K,\varepsilon}(D\beta) = \frac{1}{2\epsilon} \|D\beta+\alpha\|^2_2- \frac{\epsilon}{2} \|(D\beta+\alpha)/\epsilon - P_K((D\beta+\alpha)/\epsilon)\|^2_2

        and the gradient is given by

        .. math::

           \nabla_{\beta} h_{K,\varepsilon}(D\beta+\alpha) = D'P_K((D\beta+\alpha)/\epsilon)

        If a seminorm has several atoms, then :math:`D` is a
        `stacked' version and :math:`K` is a product
        of corresponding convex sets.

        """
        self.atoms = atoms
        if 'epsilon' in keywords:
            self.epsilon = keywords['epsilon']
        else:
            self.epsilon = 0.1
        if self.epsilon <= 0:
            raise ValueError('to smooth, epsilon must be positive')
        self.primal_shape = atoms[0].primal_shape
        try:
            for atom in atoms:
                assert(atom.primal_shape == self.primal_shape)
        except:
            raise ValueError("Atoms have different primal shapes")
        self.coefs = np.zeros(self.primal_shape)

    def smooth_eval(self, beta, mode='both'):
        """
        Evaluate a smooth function and/or its gradient

        if mode == 'both', return both function value and gradient
        if mode == 'grad', return only the gradient
        if mode == 'func', return only the function value
        """

        if mode == 'both':
            objective, grad = 0, 0
            for atom in self.atoms:
                u = atom.affine_map(beta)
                ueps = u / self.epsilon
                projected_ueps = atom.dual_prox(ueps)
                objective += self.epsilon / 2. * (np.linalg.norm(ueps)**2 - np.linalg.norm(ueps-projected_ueps)**2)
                grad += atom.adjoint_map(projected_ueps)
            return objective, grad
        elif mode == 'grad':
            grad = 0
            for atom in self.atoms:
                u = atom.affine_map(beta)
                ueps = u / self.epsilon
                projected_ueps = atom.dual_prox(ueps)
                grad += atom.adjoint_map(projected_ueps)
            return grad 
        elif mode == 'func':
            objective = 0
            for atom in self.atoms:
                u = atom.affine_map(beta)
                ueps = u / self.epsilon
                projected_ueps = atom.dual_prox(ueps)
                objective += self.epsilon / 2. * (np.linalg.norm(ueps)**2 - np.linalg.norm(ueps-projected_ueps)**2)
            return objective 
        else:
            raise ValueError("mode incorrectly specified")

