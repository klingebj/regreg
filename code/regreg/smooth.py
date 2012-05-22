import numpy as np
from scipy import sparse
import warnings
import inspect

from .composite import smooth as smooth_composite
from .affine import affine_transform, linear_transform
from .identity_quadratic import identity_quadratic

class smooth_atom(smooth_composite):

    """
    A class for representing a smooth function and its gradient
    """

    objective_template = r'''f(%(var)s)'''
    _doc_dict = {'objective': '',
                 'shape':'p',
                 'var':r'x'}

    def __init__(self, primal_shape, coef=1, offset=None,
                 quadratic=None, initial=None):
        smooth_composite.__init__(self, primal_shape,
                                  offset=offset,
                                  quadratic=quadratic,
                                  initial=initial)
        self.coef = coef
        if coef < 0:
            raise ValueError('coefs must be nonnegative to ensure convexity (assuming all atoms are indeed convex)')
        self.coefs = np.zeros(self.primal_shape)

    def smooth_objective(self, x, mode='both', check_feasibility=False):
        raise NotImplementedError
    
    @classmethod
    def affine(cls, linear_operator, offset, coef=1, diag=False,
               quadratic=None, **kws):
        """
        Keywords given in kws are passed to cls constructor along with other arguments
        """
        if not isinstance(linear_operator, affine_transform):
            l = linear_transform(linear_operator, diag=diag)
        else:
            l = linear_operator
        if not acceptable_init_args(cls, kws):
            raise ValueError("Invalid arguments being passed to initialize " + cls.__name__)
        
        atom = cls(l.primal_shape, coef=coef, offset=offset, quadratic=quadratic, **kws)
        
        return affine_smooth(atom, l)

    @classmethod
    def linear(cls, linear_operator, coef=1, diag=False,
               offset=None, 
               quadratic=None, **kws):
        """
        Keywords given in kws are passed to cls constructor along with other arguments
        """
        if not acceptable_init_args(cls, kws):
            raise ValueError("Invalid arguments being passed to initialize " + cls.__name__)

        atransform = affine_transform(linear_operator, None, diag=diag)
        atom = cls(atransform.primal_shape, coef=coef, quadratic=quadratic, offset=offset, **kws)
        
        return affine_smooth(atom, atransform)

    @classmethod
    def shift(cls, offset, coef=1, quadratic=None, **kws):
        """
        Keywords given in kws are passed to cls constructor along with other arguments
        """
        if not acceptable_init_args(cls, kws):
            raise ValueError("Invalid arguments being passed to initialize " + cls.__name__)
        
        atom = cls(offset.shape, coef=coef, quadratic=quadratic, 
                   offset=offset, **kws)
        return atom

    def scale(self, obj, copy=False):
        if self.coef != 1:
            return obj * self.coef
        if copy:
            return obj.copy()
        return obj

    def get_conjugate(self):
        raise NotImplementedError('each smooth loss should implement its own get_conjugate')

    @property
    def conjugate(self):
        if not hasattr(self, "_conjugate"):
            self._conjugate = self.get_conjugate()
            self._conjugate._conjugate = self
        return self._conjugate

def acceptable_init_args(cls, proposed_keywords):
    """
    Check that the keywords in the dictionary proposed_keywords are arguments to __init__ of class cls

    Returns True/False
    """
    args = inspect.getargspec(cls.__init__).args
    forbidden = ['self', 'primal_shape', 'coef', 'quadratic', 'initial', 'offset']
    for kw in proposed_keywords.keys():
        if not kw in args:
            return False
        if kw in forbidden:
            return False
    return True

class affine_smooth(smooth_atom):

    # if smooth_obj is a class, an object is created
    # smooth_obj(*args, **keywords)
    # else, it is assumed to be an instance of smooth_function
 
    def __init__(self, smooth_atom, atransform, store_grad=True, diag=False):
        self.store_grad = store_grad
        self.sm_atom = smooth_atom
        if not isinstance(atransform, affine_transform):
            atransform = linear_transform(atransform, diag=diag)
        self.affine_transform = atransform
        self.primal_shape = atransform.primal_shape
        self.coefs = np.zeros(self.primal_shape)

    def latexify(self, var='x', idx=''):
        obj = self.sm_atom.latexify(var='D_{%s}%s' % (idx, var), idx=idx)
        if not self.quadratic.iszero:
            return ' + '.join([self.quadratic.latexify(var=var,idx=idx),obj])
        return obj

    def _get_coef(self):
        return self.sm_atom.coef

    def _set_coef(self, coef):
        self.sm_atom.coef = coef
    coef = property(_get_coef, _set_coef)

    def smooth_objective(self, x, mode='both', check_feasibility=False):
        eta = self.affine_transform.affine_map(x)
        if mode == 'both':
            v, g = self.sm_atom.smooth_objective(eta, mode='both')
            if self.store_grad:
                self.grad = g
            g = self.affine_transform.adjoint_map(g)
            return v, g
        elif mode == 'grad':
            g = self.sm_atom.smooth_objective(eta, mode='grad')
            if self.store_grad:
                self.grad = g
            g = self.affine_transform.adjoint_map(g)
            return g 
        elif mode == 'func':
            v = self.sm_atom.smooth_objective(eta, mode='func')
            return v 

#     @property
#     def composite(self):
#         initial = np.zeros(self.primal_shape)
#         return smooth_composite(self.smooth_objective, initial)

    @property
    def dual(self):
        try: 
            conj = self.sm_atom.conjugate
            return self.affine_transform, conj
        except:
            return None

    def __repr__(self):
        return ("affine_smooth(%s, %s, store_grad=%s)" % 
                (str(self.sm_atom),
                str(self.affine_transform),
                self.store_grad))

def squaredloss(linear_operator, offset, coef=1):
    # the affine method gets rid of the need for the squaredloss class
    # as previously written squared loss had a factor of 2

    #return quadratic.affine(-linear_operator, offset, coef=coef/2., initial=np.zeros(linear_operator.shape[1]))
    return quadratic.affine(linear_operator, -offset, coef=coef/2.)

def signal_approximator(offset, coef=1):
    return quadratic.shift(-offset, coef)

class zero(smooth_atom):

    def smooth_objective(self, x, mode='both', check_feasibility=False):
        if mode == 'both':
            return 0., np.zeros(x.shape)
        elif mode == 'func':
            return 0.
        elif mode == 'grad':
            return np.zeros(x.shape)
        raise ValueError("Mode not specified correctly")

class logistic_deviance(smooth_atom):

    """
    A class for combining the logistic log-likelihood with a general seminorm
    """

    objective_template = r"""\ell^{L}\left(%(var)s\right)"""
    #TODO: Make init more standard, replace np.dot with shape friendly alternatives in case successes.shape is (n,1)

    def __init__(self, primal_shape, successes, 
                 trials=None, coef=1., offset=None,
                 quadratic=None,
                 initial=None):

        smooth_atom.__init__(self,
                             primal_shape,
                             offset=offset,
                             quadratic=quadratic,
                             initial=initial)

        if sparse.issparse(successes):
            #Convert sparse success vector to an array
            self.successes = successes.toarray().flatten()
        else:
            self.successes = np.asarray(successes)

        if trials is None:
            if not set([0,1]).issuperset(np.unique(self.successes)):
                raise ValueError("Number of successes is not binary - must specify number of trials")
            self.trials = np.ones(self.successes.shape, np.float)
        else:
            if np.min(trials-self.successes) < 0:
                raise ValueError("Number of successes greater than number of trials")
            if np.min(self.successes) < 0:
                raise ValueError("Response coded as negative number - should be non-negative number of successes")
            self.trials = trials * 1.

        saturated = self.successes / self.trials
        deviance_terms = np.log(saturated) * self.successes + np.log(1-saturated) * (self.trials - self.successes)
        deviance_constant = -2 * coef * deviance_terms[~np.isnan(deviance_terms)].sum()

        devq = identity_quadratic(0,0,0,-deviance_constant)
        self.quadratic += devq

    def smooth_objective(self, x, mode='both', check_feasibility=False):
        """
        Evaluate a smooth function and/or its gradient

        if mode == 'both', return both function value and gradient
        if mode == 'grad', return only the gradient
        if mode == 'func', return only the function value
        """
        
        #Check for overflow in np.exp (can occur during initial backtracking steps)
        x = self.apply_offset(x)
        if np.max(x) > 1e2:
            overflow = True
            not_overflow_ind = np.where(x <= 1e2)[0]
            exp_x = np.exp(x[not_overflow_ind])
        else:
            overflow = False
            exp_x = np.exp(x)
            
        if mode == 'both':
            ratio = self.trials * 1.
            if overflow:
                log_exp_x = x * 1.
                log_exp_x[not_overflow_ind] = np.log(1.+exp_x)
                ratio[not_overflow_ind] *= exp_x/(1.+exp_x)
            else:
                log_exp_x = np.log(1.+exp_x)
                ratio *= exp_x/(1.+exp_x)
                
            f, g = -2 * self.scale((np.dot(self.successes,x) - np.sum(self.trials*log_exp_x))), -2 * self.scale(self.successes-ratio)
            return f, g
        elif mode == 'grad':
            ratio = self.trials * 1.
            if overflow:
                ratio[not_overflow_ind] *= exp_x/(1.+exp_x)
            else:
                ratio *= exp_x/(1.+exp_x)
            f, g = None, - 2 * self.scale(self.successes-ratio)
            return g
        elif mode == 'func':
            if overflow:
                log_exp_x = x * 1.
                log_exp_x[not_overflow_ind] = np.log(1.+exp_x)
            else:
                log_exp_x = np.log(1.+exp_x)
            f, g = -2 * self.scale(np.dot(self.successes,x) - np.sum(self.trials * log_exp_x)), None
            return f
        else:
            raise ValueError("mode incorrectly specified")


class poisson_deviance(smooth_atom):

    """
    A class for combining the Poisson log-likelihood with a general seminorm
    """

    objective_template = r"""\ell^{P}\left(%(var)s\right)"""

    def __init__(self, primal_shape, counts, coef=1., offset=None,
                 quadratic=None,
                 initial=None):

        smooth_atom.__init__(self,
                             primal_shape,
                             offset=offset,
                             quadratic=quadratic,
                             initial=initial)

        if sparse.issparse(counts):
            #Convert sparse success vector to an array
            self.counts = counts.toarray().flatten()
        else:
            self.counts = counts

        if not np.allclose(np.round(self.counts),self.counts):
            raise ValueError("Counts vector is not integer valued")
        if np.min(self.counts) < 0:
            raise ValueError("Counts vector is not non-negative")

        saturated = counts
        deviance_terms = -2 * coef * ((counts - 1) * np.log(counts))
        deviance_terms[counts == 0] = 0

        deviance_constant = -2 * coef * deviance_terms.sum()

        devq = identity_quadratic(0,0,0,-deviance_constant)
        self.quadratic += devq

    def smooth_objective(self, x, mode='both', check_feasibility=False):
        """
        Evaluate a smooth function and/or its gradient

        if mode == 'both', return both function value and gradient
        if mode == 'grad', return only the gradient
        if mode == 'func', return only the function value
        """
        x = self.apply_offset(x)
        exp_x = np.exp(x)
        
        if mode == 'both':
            f, g = -2. * self.scale(-np.sum(exp_x) + np.dot(self.counts,x)), -2. * self.scale(self.counts - exp_x)
            return f, g
        elif mode == 'grad':
            f, g = None, -2. * self.scale(self.counts - exp_x)
            return g
        elif mode == 'func':
            f, g =  -2. * self.scale(-np.sum(exp_x) + np.dot(self.counts,x)), None
            return f
        else:
            raise ValueError("mode incorrectly specified")


class multinomial_deviance(smooth_atom):

    """
    A class for baseline-category logistic regression for nominal responses (e.g. Agresti, pg 267)
    """

    objective_template = r"""\ell^{M}\left(%(var)s\right)"""

    def __init__(self, primal_shape, counts, coef=1., offset=None,
                 quadratic=None):

        smooth_atom.__init__(self,
                             primal_shape,
                             offset=offset,
                             quadratic=quadratic,
                             initial=initial)

        if sparse.issparse(counts):
            #Convert sparse success vector to an array
            self.counts = counts.toarray()
        else:
            self.counts = counts

        self.J = self.counts.shape[1]
        #Select the counts for the first J-1 categories
        self.firstcounts = self.counts[:,range(self.J-1)]

        if not np.allclose(np.round(self.counts),self.counts):
            raise ValueError("Counts vector is not integer valued")
        if np.min(self.counts) < 0:
            raise ValueError("Counts vector is not non-negative")

        self.trials = np.sum(self.counts, axis=1)

        if primal_shape[1] != self.J - 1:
            raise ValueError("Primal shape is incorrect - should only have coefficients for first J-1 categories")

        saturated = self.counts / (1. * self.trials[:,np.newaxis])
        deviance_terms = np.log(saturated) * self.counts
        deviance_terms[np.isnan(deviance_terms)] = 0
        deviance_constant = -2 * coef * deviance_terms.sum()

        devq = identity_quadratic(0,0,0,-deviance_constant)
        self.quadratic += devq

    def smooth_objective(self, x, mode='both', check_feasibility=False):
        """
        Evaluate a smooth function and/or its gradient

        if mode == 'both', return both function value and gradient
        if mode == 'grad', return only the gradient
        if mode == 'func', return only the function value
        """
        x = self.apply_offset(x)
        exp_x = np.exp(x)

        #TODO: Using transposes to scale the rows of a 2d array - should we use an affine_transform to do this?
        #JT: should be able to do this with np.newaxis

        if mode == 'both':
            ratio = ((self.trials/(1. + np.sum(exp_x, axis=1))) * exp_x.T).T
            f, g = -2. * self.scale(np.sum(self.firstcounts * x) -  np.dot(self.trials, np.log(1. + np.sum(exp_x, axis=1)))), - 2 * self.scale(self.firstcounts - ratio) 
            return f, g
        elif mode == 'grad':
            ratio = ((self.trials/(1. + np.sum(exp_x, axis=1))) * exp_x.T).T
            f, g = None, - 2 * self.scale(self.firstcounts - ratio) 
            return g
        elif mode == 'func':
            f, g = -2. * self.scale(np.sum(self.firstcounts * x) -  np.dot(self.trials, np.log(1. + np.sum(exp_x, axis=1)))), None
            return f
        else:
            raise ValueError("mode incorrectly specified")


