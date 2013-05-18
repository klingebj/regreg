from numpy.linalg import norm
from numpy import zeros, array, any as npany
import new
from copy import copy

# local imports

from ..identity_quadratic import identity_quadratic as sq
from ..algorithms import FISTA

from ..objdoctemplates import objective_doc_templater
from ..doctemplates import (doc_template_user, doc_template_provider)

@objective_doc_templater()
class composite(object):
    """
    A generic way to specify a problem in composite form.
    """

    objective_template = r"""f(%(var)s)"""
    objective_vars = {'var': r'\beta', 'shape':'p', 'offset':r'\alpha'}

    def __init__(self, shape, offset=None,
                 quadratic=None, initial=None):

        self.offset = offset
        if offset is not None:
            self.offset = array(offset)

        if type(shape) == type(1):
            self.shape = (shape,)
        else:
            self.shape = shape

        if quadratic is not None:
            self.quadratic = quadratic
        else:
            self.quadratic = sq(0,0,0,0)

        if initial is None:
            self.coefs = zeros(self.shape)
        else:
            self.coefs = initial.copy()

    def latexify(self, var=None, idx=''):
        template_dict = self.objective_vars.copy()
        template_dict['idx'] = idx
        if var is not None:
            template_dict['var'] = var
        if hasattr(self, 'offset') and self.offset is not None and npany(self.offset != 0):
            template_dict['var'] = var + (r' - %(offset)s_{%(idx)s}' % template_dict)

        obj = self.objective_template % template_dict
        template_dict['obj'] = obj

        if not self.quadratic.iszero:
            return ' + '.join([obj, self.quadratic.latexify(var=var, idx=idx)])
        return obj

    def _repr_latex_(self):
        return r'$$' + self.latexify() + r'$$'

    def nonsmooth_objective(self, x, check_feasibility=False):
        return self.quadratic.objective(x, 'func')

    def smooth_objective(self, x, mode='both', check_feasibility=False):
        '''
        The smooth_objective and the quadratic_objective combined.
        '''
        raise NotImplementedError

    def objective(self, x, check_feasibility=False):
        return self.smooth_objective(x,mode='func', check_feasibility=check_feasibility) + self.nonsmooth_objective(x, check_feasibility=check_feasibility)

    @doc_template_provider
    def proximal_optimum(self, quadratic):
        r"""
        Returns

        .. math::

           \inf_{x \in \mathbb{R}^p} Q(x)
            + h(x)

        where $p$ = ``x.shape[0]``, $Q(x)=$ `self.quadratic` and

        .. math::
        
            h(%(var)s) = %(ns_objective)s

        """
        argmin = self.proximal(quadratic)
        if self.quadratic is None:
            return argmin, lipschitz * norm(x-argmin)**2 / 2. + self.nonsmooth_objective(argmin)  
        else:
            return argmin, lipschitz * norm(x-argmin)**2 / 2. + self.nonsmooth_objective(argmin) + self.quadratic.objective(argmin, 'func') 

    def proximal_step(self, quadratic, prox_control=None):
        """
        Compute the proximal optimization

        prox_control: If not None, then a dictionary of parameters for the prox procedure
        """
        # This seems like a null op -- if all proximals accept optional prox_control
        if prox_control is None:
            return self.proximal(quadratic)
        else:
            return self.proximal(quadratic, prox_control=prox_control)

    def apply_offset(self, x):
        """
        If self.offset is not None, return x-self.offset, else return x.
        """
        if self.offset is not None:
            return x - self.offset
        return x

    def set_quadratic(self, quadratic):
        """
        Set the quadratic part of the composite.
        """
        self._quadratic = quadratic

    def get_quadratic(self):
        """
        Get the quadratic part of the composite.
        """
        if not hasattr(self, "_quadratic"):
            self._quadratic = sq(None, None, None, None)
        return self._quadratic
    quadratic = property(get_quadratic, set_quadratic)

    def smoothed(self, smoothing_quadratic):
        '''
        Add quadratic smoothing term
        '''
        conjugate_atom = copy(self.conjugate)
        sq = smoothing_quadratic
        if sq.coef in [None, 0]:
            raise ValueError('quadratic term of smoothing_quadratic must be non 0')
        total_q = sq

        if conjugate_atom.quadratic is not None:
            total_q = sq + conjugate_atom.quadratic
        conjugate_atom.set_quadratic(total_q)
        smoothed_atom = conjugate_atom.conjugate
        return smoothed_atom

    def solve(self, quadratic=None, return_optimum=False, **fit_args):
        raise NotImplementedError('subclasses must implement their own solve methods')

@objective_doc_templater()
class nonsmooth(composite):
    """
    A composite subclass that explicitly returns 0
    as smooth_objective.
    """

    def smooth_objective(self, x, mode='both', check_feasibility=False):
        if mode == 'both':
            return 0., zeros(x.shape)
        elif mode == 'func':
            return 0.
        elif mode == 'grad':
            return zeros(x.shape)
        raise ValueError("Mode not specified correctly")

    def solve(self, quadratic=None, return_optimum=False, **fit_args):
        if quadratic is None:
            quadratic = sq(0,0,0,0)
        self.coefs = self.proximal(quadratic)
        if return_optimum:
            return self.objective(self.coefs) + quadratic.objective(self.coefs, 'func'), self.coefs
        else:
            return self.coefs

@objective_doc_templater()
class smooth(composite):

    """
    A composite subclass that has 0 as 
    nonsmooth_objective and the proximal
    is a null-op.
    """

    objective_vars = composite.objective_vars.copy()
    objective_vars['coef'] = 'C'

    def get_lipschitz(self):
        if hasattr(self, '_lipschitz'):
            return self._lipschitz + self.quadratic.coef
        return self.quadratic.coef

    def set_lipschitz(self, value):
        if value < 0:
            raise ValueError('Lipschitz constant must be non-negative')
        self._lipschitz = value
    lipschitz = property(get_lipschitz, set_lipschitz)

    def smooth_objective(self, x, mode='func', check_feasibility=False):
        return self._smooth_objective(x, mode=mode, check_feasibility=check_feasibility)

    def proximal(self, quadratic):
        totalq = self.quadratic + quadratic
        return -totalq.linear_term / totalq.coef

    def solve(self, quadratic=None, return_optimum=False, **fit_args):
        if quadratic is None:
            quadratic = sq(0,0,0,0)
        oldq, self.quadratic = self.quadratic, self.quadratic + quadratic
        self.solver = FISTA(self)
        self.solver.fit(**fit_args)
        self.quadratic = oldq

        if return_optimum:
            return self.objective(self.coefs), self.coefs
        else:
            return self.coefs

class smooth_conjugate(smooth):

    def __init__(self, atom, quadratic=None):
        """
        Given an atom,
        compute the conjugate of this atom plus 
        an identity_quadratic which will be 
        a smooth version of the conjugate of the atom.

        should we have an argument "collapse" that makes a copy?

        """
        # this holds a pointer to the original atom,
        # but will be replaced later

        self.atom = atom
        if quadratic is None:
            quadratic = sq(0,0,0,0)
        self.smoothing_quadratic = quadratic
        self.total_quadratic = self.atom.quadratic + self.smoothing_quadratic

        if self.total_quadratic.coef in [0,None]:
            raise ValueError('the atom must have non-zero quadratic term to compute ensure smooth conjugate')

        self.shape = atom.shape

        # A smooth conjugate is the conjugate of some $f$ with an identity quadratic added to it, or
        # $$
        # h(u) = \sup_x \left( u^Tx - \frac{\kappa}{2} \|x\|^2_2 - \beta^Tx-c-f(x) \right).
        # $$
        # Suppose we add a quadratic to $h$ to get
        # $$
        # \tilde{h}(u) = \frac{r}{2} \|u\|^2_2 + u^T\gamma + a + h(u)$$
        # and take the conjugate again:
        # $$
        # \begin{aligned}
        # g(v) &= \sup_{u} u^Tv - \tilde{h}(u) \\
        # &= \sup_u u^Tv -  \frac{r}{2} \|u\|^2_2 - u^T\gamma-a - h(u) \\
        # &=  \sup_u u^Tv - \frac{r}{2} \|u\|^2_2 - u^T\gamma-a  - \sup_x \left( u^Tx - \frac{\kappa}{2} \|x\|^2_2 - \beta^Tx-c-f(x)  \right)\\
        # &= \sup_u u^Tv - \frac{r}{2} \|u\|^2_2 - u^T\gamma-a + \inf_x \left(  \frac{\kappa}{2} \|x\|^2_2  +\beta^Tx + c +f(x) - u^Tx  \right)\\
        # &= \sup_u \inf_x u^Tv - \frac{r}{2} \|u\|^2_2 - u^T\gamma-a +  \frac{\kappa}{2} \|x\|^2_2 + \beta^Tx + c +f(x) - u^Tx \\
        # &=  \inf_x \sup_u u^Tv - \frac{r}{2} \|u\|^2_2 - u^T\gamma-a +  \frac{\kappa}{2} \|x\|^2_2  + \beta^Tx + c +f(x) - u^Tx \\
        # &=  \inf_x \sup_u \left(u^Tv - \frac{r}{2} \|u\|^2_2 - u^T\gamma- u^Tx\right)-a +  \frac{\kappa}{2} \|x\|^2_2 +   \beta^Tx + c +f(x)  \\
        # &=  \inf_x \frac{1}{2r} \|x+\gamma-v\|^2_2 -a +  \frac{\kappa}{2} \|x\|^2_2 + \beta^Tx + c +f(x)  \\
        # &= c-a + \frac{1}{2r} \|\gamma-v\|^2_2 - \sup_x \left((v/r)^Tx - \left(\frac{1}{r} + \kappa\right) \|x\|^2_2 - x^T(\beta+\gamma/r) - f(x) \right) \\
        # \end{aligned}
        # $$

        # This says that for $r > 0$ the conjugate of a smooth conjugate with a quadratic added to it is a quadratic plus a modified smooth conjugate evaluated at $v/r$.

        # What if $r=0$? Well,
        # then 
        # $$
        # \begin{aligned}
        # g(v) &= \sup_{u} u^Tv - \tilde{h}(u) \\
        # &= \sup_u u^Tv  - u^T\gamma-a - h(u) \\
        # &=  \sup_u u^Tv  - u^T\gamma-a  - \sup_x \left( u^Tx - \frac{\kappa}{2} \|x\|^2_2 - \beta^Tx-c-f(x)  \right)\\
        # &= \sup_u u^Tv - u^T\gamma-a + \inf_x \left(  \frac{\kappa}{2} \|x\|^2_2  +\beta^Tx + c +f(x) - u^Tx  \right)\\
        # &= \sup_u \inf_x u^Tv - u^T\gamma-a +  \frac{\kappa}{2} \|x\|^2_2 + \beta^Tx + c +f(x) - u^Tx \\
        # &= \inf_x \sup_u u^Tv - u^T\gamma-a +  \frac{\kappa}{2} \|x\|^2_2 + \beta^Tx + c +f(x) - u^Tx \\
        # &=   \frac{\kappa}{2} \|v-\gamma\|^2_2 + \beta^T(v-\gamma) + c-a +f(v-\gamma) \\
        # \end{aligned}
        # $$
        # where, in the last line we have used the fact that the $\sup$ over $u$ in the second to last line is infinite unless $x=v-\gamma$.

    def get_conjugate(self):
        if self.quadratic.iszero:
            if self.smoothing_quadratic.iszero:
                return self.atom
            else:
                atom = copy(self.atom)
                atom.quadratic = atom.quadratic + self.smoothing_quadratic
                return atom
        else:
            q = self.quadratic.collapsed()
            if q.coef == 0:
                newq = copy(atom.quadratic)
                newq.constant_term -= q.constant_term
                offset = -q.linear_term
                if atom.offset is not None:
                    offset -= atom.offset
                atom = copy(atom)
                atom.offset = -offset
                atom.quadratic=newq
                return atom
            if q.coef != 0:
                r = q.coef
                sq_ = self.smoothing_quadratic
                newq = sq_ + q.conjugate
                new_smooth = smooth_conjugate(self.atom, quadratic=newq)
                output = smooth(self.atom.shape,
                                offset=None,
                                quadratic=sq(1./r, q.linear_term, 0, 0),
                                initial=None)
                output.smoothed_atom = new_smooth

                def smooth_objective(self, x, mode='func', check_feasibility=False):
                    # what if self.quadratic is later changed? hmm..
                    r = 1. / self.quadratic.coef
                    if mode == 'func':
                        v = self.smoothed_atom.smooth_objective(x/r, mode=mode, 
                                                                check_feasibility=check_feasibility)
                        return self.smoothing_quadratic.objective(x, 'func') - v
                    elif mode == 'both':
                        v1, g1 = self.smoothed_atom.smooth_objective(x/r, mode=mode, 
                                                                     check_feasibility=check_feasibility)
                        v2, g2 = self.smoothing_quadratic.objective(x, mode=mode, 
                                                                    check_feasibility=check_feasibility)
                        return v2-v1, g2-g1/r
                    elif mode == 'grad':
                        g1 = self.smoothed_atom.smooth_objective(x/r, mode='grad', 
                                                                     check_feasibility=check_feasibility)
                        g2 = self.smoothing_quadratic.objective(x, mode='grad', 
                                                                    check_feasibility=check_feasibility)
                        return g2-g1/r
                    else:
                        raise ValueError("mode incorrectly specified")
                        
                output.smooth_objective = type(output.smooth_objective)(smooth_objective,
                                                                        output, 
                                                                        smooth)
                return output
    conjugate = property(get_conjugate)

    def latexify(self, var=None, idx=''):
        template_dict = self.atom.objective_vars.copy()
        template_dict['idx'] = idx
        if var is not None:
            template_dict['var'] = var

        template_dict['f'] = self.atom.latexify(var='u', idx=idx)

        objective = r' \sup_{u \in \mathbb{R}^{%(shape)s} } \left[ \langle %(var)s, u \rangle - \left(%(f)s \right) \right]' % template_dict
        return objective

    def __repr__(self):
        return 'smooth_conjugate(%s,%s)' % (repr(self.atom), repr(self.quadratic))

    def smooth_objective(self, x, mode='both', check_feasibility=False):
        """
        Evaluate a smooth function and/or its gradient

        if mode == 'both', return both function value and gradient
        if mode == 'grad', return only the gradient
        if mode == 'func', return only the function value
        """

        q = self.smoothing_quadratic + sq(0,0,-x,0) 

        if mode == 'both':
            optimal_value, argmin = self.atom.solve(quadratic=q, return_optimum=True)
            objective = -optimal_value
            # retain a reference
            self.grad = argmin
            return objective, self.grad
        elif mode == 'grad':
            argmin = self.atom.solve(quadratic=q)
            # retain a reference
            self.grad = argmin
            return self.grad
        elif mode == 'func':
            optimal_value, argmin = self.atom.solve(quadratic=q, return_optimum=True)
            objective = -optimal_value
            # retain a reference
            self.grad = argmin
            return objective
        else:
            raise ValueError("mode incorrectly specified")

    def proximal(self, proxq, prox_control=None):
        raise ValueError('no proximal defined')
