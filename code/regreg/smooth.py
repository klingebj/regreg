import numpy as np
from scipy import sparse
from affine import affine_transform
import warnings
from composite import smooth as smooth_composite

# class smooth_function(smooth_composite):
#     """
#     A container class for smooth_atom classes
#     """

#     # TODO? use a list for atoms instead of *atoms?
#     def __init__(self, *atoms, **keywords):
#         # why do we have this -- XXX
#         if not set(keywords.keys()).issubset(['coef', 'constant_term']):
#             warnings.warn('only keyword argument should be multiplier, "coef" and "constant", got %s' % `keywords`)

#         self.coef = 1
#         if keywords.has_key('coef'):
#             self.coef = keywords['coef']

#         self.constant_term = 0
#         if keywords.has_key('constant_term'):
#             self.constant_term = keywords['constant_term']

#         self.atoms = []
#         self.primal_shape = -1
#         for atom in atoms:
#             if self.primal_shape == -1:
#                 self.primal_shape = atom.primal_shape
#             else:
#                 if atom.primal_shape != self.primal_shape:
#                     raise ValueError("primal dimensions don't agree")
#             self.atoms.append(atom)
#         self.coefs = np.zeros(self.primal_shape)

#     def smooth_objective(self, x, mode='both', check_feasibility=False):
#         """
#         Evaluate a smooth function and/or its gradient

#         if mode == 'both', return both function value and gradient
#         if mode == 'grad', return only the gradient
#         if mode == 'func', return only the function value
#         """

#         if mode == 'func':
#             if len(self.atoms) > 1:
#                 v = 0.
#                 for atom in self.atoms:
#                     v += atom.smooth_objective(x, mode=mode)
#                 return self.scale(v) + self.constant_term
#             else:
#                 return self.scale(self.atoms[0].smooth_objective(x, mode=mode)) + self.constant_term
#         elif mode == 'grad':
#             if len(self.atoms) > 1:
#                 g = np.zeros(self.primal_shape)
#                 for atom in self.atoms:
#                     g += atom.smooth_objective(x, mode=mode)
#                 return self.scale(g)
#             else:
#                 return self.scale(self.atoms[0].smooth_objective(x, mode=mode))
#         elif mode == 'both':
#             if len(self.atoms) > 1:
#                 v = 0.
#                 g = np.zeros(self.primal_shape)
#                 for atom in self.atoms:
#                     output = atom.smooth_objective(x, mode=mode)
#                     v += output[0]
#                     g += output[1]
#                 return self.scale(v) + self.constant_term, self.scale(g)
#             else:
#                 v, g = self.atoms[0].smooth_objective(x, mode=mode)
#                 return self.scale(v) + self.constant_term, self.scale(g)
#         else:
#             raise ValueError("Mode specified incorrectly")


class smooth_atom(smooth_composite):

    """
    A class for representing a smooth function and its gradient
    """

    def __init__(self, primal_shape, coef=1, constant_term=0):
        self.constant_term = constant_term
        self.primal_shape = primal_shape
        self.coef = coef
        if coef < 0:
            raise ValueError('coefs must be nonnegative to ensure convexity (assuming all atoms are indeed convex)')
        self.coefs = np.zeros(self.primal_shape)
        raise NotImplementedError

    def smooth_objective(self, x, mode='both', check_feasibility=False):
        raise NotImplementedError
    
    @classmethod
    def affine(cls, linear_operator, offset, coef=1, diag=False,
               constant_term=0):
        """
        Args and keywords passed to cls constructor along with
        l and primal_shape
        """
        atransform = affine_transform(linear_operator, offset, diag=diag)
        atom = cls(atransform.primal_shape, coef=coef, constant_term=constant_term)
        
        return affine_smooth(atom, atransform)

    @classmethod
    def linear(cls, linear_operator, coef=1, diag=False,
               constant_term=0):
        """
        Args and keywords passed to cls constructor along with
        l and primal_shape
        """
        atransform = affine_transform(linear_operator, None, diag=diag)
        atom = cls(atransform.primal_shape, coef=coef, constant_term=constant_term)
        
        return affine_smooth(atom, atransform)

    @classmethod
    def shift(cls, offset, coef=1, 
              constant_term=0,
              args=(), keywords={}):
        """
        Args and keywords passed to cls constructor along with
        l and primal_shape
        """
        atransform = affine_transform(None, offset)
        atom = cls(atransform.primal_shape, coef=coef, constant_term=constant_term)
        return affine_smooth(atom, atransform)

    def scale(self, obj, copy=False):
        if self.coef != 1:
            return obj * self.coef
        if copy:
            return obj.copy()
        return obj
    
class affine_smooth(smooth_atom):

    # if smooth_obj is a class, an object is created
    # smooth_obj(*args, **keywords)
    # else, it is assumed to be an instance of smooth_function
 
    def __init__(self, smooth_atom, atransform):
        self.sm_atom = smooth_atom
        self.affine_transform = atransform
        self.primal_shape = self.affine_transform.primal_shape
        self.coefs = np.zeros(self.primal_shape)

    def _get_coef(self):
        return self.sm_atom.coef

    def _set_coef(self, coef):
        self.sm_atom.coef = coef
    coef = property(_get_coef, _set_coef)

    def smooth_objective(self, x, mode='both', check_feasibility=False):
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

def squaredloss(linear_operator, offset, coef=1):
    # the affine method gets rid of the need for the squaredloss class
    # as previously written squared loss had a factor of 2

    #return l2normsq.affine(-linear_operator, offset, coef=coef/2., initial=np.zeros(linear_operator.shape[1]))
    return l2normsq.affine(-linear_operator, offset, coef=coef/2.)

class l2normsq(smooth_atom):
    """
    The square of the l2 norm
    """

    def __init__(self, primal_shape, coef=None, Q=None, Qdiag=False,
                 constant_term=0):
        self.constant_term = constant_term
        self.Q = Q
        if self.Q is not None:
            self.Q_transform = affine_transform(Q, 0, Qdiag)
        if type(primal_shape) == type(1):
            self.primal_shape = (primal_shape,)
        else:
            self.primal_shape = primal_shape
        self.coef = coef

    def smooth_objective(self, x, mode='both', check_feasibility=False):
        """
        Evaluate a smooth function and/or its gradient

        if mode == 'both', return both function value and gradient
        if mode == 'grad', return only the gradient
        if mode == 'func', return only the function value
        """

        if self.Q is None:
            if mode == 'both':
                return self.scale(np.linalg.norm(x)**2) + self.constant_term, self.scale(2 * x)
            elif mode == 'grad':
                return self.scale(2 * x)
            elif mode == 'func':
                return self.scale(np.linalg.norm(x)**2) + self.constant_term
            else:
                raise ValueError("mode incorrectly specified")
        else:
            if mode == 'both':
                return self.scale(np.sum(x * self.Q_transform.linear_map(x))) + self.constant_term, self.scale(2 * self.Q_transform.linear_map(x))
            elif mode == 'grad':
                return self.scale(2 * self.Q_transform.linear_map(x))
            elif mode == 'func':
                return self.scale(np.sum(x * self.Q_transform.linear_map(x))) + self.constant_term
            else:
                raise ValueError("mode incorrectly specified")
            
class linear(smooth_atom):

    def __init__(self, vector, coef=1):
        self.vector = coef * vector
        self.primal_shape = vector.shape
        self.coef =1
        self.coefs = np.zeros(self.primal_shape)

    def smooth_objective(self, x, mode='both', check_feasibility=False):
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
    

def signal_approximator(offset, coef=1):
    return l2normsq.shift(-offset, coef)

class logistic_loglikelihood(smooth_atom):

    """
    A class for combining the logistic log-likelihood with a general seminorm
    """

    #TODO: Make init more standard, replace np.dot with shape friendly alternatives in case successes.shape is (n,1)

    def __init__(self, linear_operator, successes, trials=None, offset=None, coef=1):
        self.affine_transform = affine_transform(linear_operator, offset)

        if sparse.issparse(successes):
            #Convert sparse success vector to an array
            self.successes = successes.toarray().flatten()
        else:
            self.successes = successes

        if trials is None:
            if not set([0,1]).issuperset(np.unique(self.successes)):
                raise ValueError("Number of successes is not binary - must specify number of trials")
            self.trials = np.ones(self.successes.shape)
        else:
            if np.min(trials-self.successes) < 0:
                raise ValueError("Number of successes greater than number of trials")
            if np.min(self.successes) < 0:
                raise ValueError("Response coded as negative number - should be non-negative number of successes")
            self.trials = trials
        
        self.primal_shape = self.affine_transform.primal_shape
        self.coef = coef

    def smooth_objective(self, x, mode='both', check_feasibility=False):
        """
        Evaluate a smooth function and/or its gradient

        if mode == 'both', return both function value and gradient
        if mode == 'grad', return only the gradient
        if mode == 'func', return only the function value
        """
        
        yhat = self.affine_transform.affine_map(x)

        #Check for overflow in np.exp (can occur during initial backtracking steps)
        if np.max(yhat) > 1e2:
            overflow = True
            not_overflow_ind = np.where(yhat <= 1e2)[0]
            exp_yhat = np.exp(yhat[not_overflow_ind])
        else:
            overflow = False
            exp_yhat = np.exp(yhat)

            
        if mode == 'both':
            ratio = self.trials * 1.
            if overflow:
                log_exp_yhat = yhat * 1.
                log_exp_yhat[not_overflow_ind] = np.log(1.+exp_yhat)
                ratio[not_overflow_ind] *= exp_yhat/(1.+exp_yhat)
            else:
                log_exp_yhat = np.log(1.+exp_yhat)
                ratio *= exp_yhat/(1.+exp_yhat)
                
            return -2 * self.scale((np.dot(self.successes,yhat) - np.sum(self.trials*log_exp_yhat))), -2 * self.scale(self.affine_transform.adjoint_map(self.successes-ratio))

        elif mode == 'grad':
            ratio = self.trials * 1.
            if overflow:
                ratio[not_overflow_ind] *= exp_yhat/(1.+exp_yhat)
            else:
                ratio *= exp_yhat/(1.+exp_yhat)
            return - 2 * self.scale(self.affine_transform.adjoint_map(self.successes-ratio))

        elif mode == 'func':
            if overflow:
                log_exp_yhat = yhat * 1.
                log_exp_yhat[not_overflow_ind] = np.log(1.+exp_yhat)
            else:
                log_exp_yhat = np.log(1.+exp_yhat)
            return -2 * self.scale(np.dot(self.successes,yhat) - np.sum(self.trials * log_exp_yhat))
        else:
            raise ValueError("mode incorrectly specified")

# class zero(smooth_function):

#     def __init__(self, primal_shape):
#         self.primal_shape = primal_shape

#     def smooth_objective(self, x, mode='both', check_feasibility=False):
#         """
#         Evaluate a smooth function and/or its gradient

#         if mode == 'both', return both function value and gradient
#         if mode == 'grad', return only the gradient
#         if mode == 'func', return only the function value
#         """
        
#         if mode == 'both':
#             return 0, np.zeros(x.shape)
#         elif mode == 'grad':
#             return np.zeros(x.shape)
#         elif mode == 'func':
#             return 0
#         else:
#             raise ValueError("mode incorrectly specified")


