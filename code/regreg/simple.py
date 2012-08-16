"""
This module has a class for specifying a problem from just
a smooth function and a single penalty.

"""
import numpy as np

from .composite import composite
from .affine import identity, scalar_multiply, astransform, adjoint
from .atoms import atom
from .cones import zero as zero_cone
from .smooth import zero as zero_smooth, sum as smooth_sum, affine_smooth
from .identity_quadratic import identity_quadratic
from .algorithms import FISTA

class simple_problem(composite):
    
    def __init__(self, smooth_atom, proximal_atom):
        self.smooth_atom = smooth_atom
        self.proximal_atom = proximal_atom
        self.coefs = self.smooth_atom.coefs = self.proximal_atom.coefs

    def smooth_objective(self, x, mode='both', check_feasibility=False):
        """
        This class explicitly assumes that
        the proximal_atom has 0 for smooth_objective.
        """
        vs = self.smooth_atom.smooth_objective(x, mode, check_feasibility)
        return vs

    def nonsmooth_objective(self, x, check_feasibility=False):
        vn = self.proximal_atom.nonsmooth_objective(x, check_feasibility=check_feasibility)
        vs = self.smooth_atom.nonsmooth_objective(x, check_feasibility=check_feasibility)
        return vn + vs + self.quadratic.objective(x, 'func')

    def proximal(self, proxq):
        proxq = proxq + self.smooth_atom.quadratic + self.quadratic
        return self.proximal_atom.solve(proxq)

    @staticmethod
    def smooth(smooth_atom):
        """
        A problem with no nonsmooth part except possibly
        the quadratic of smooth_atom.

        The proximal function is (almost) a nullop.
        """
        proximal_atom = zero_cone(smooth_atom.primal_shape)
        return simple_problem(smooth_atom, proximal_atom)

    @staticmethod
    def nonsmooth(proximal_atom):
        """
        A problem with no nonsmooth part except possibly
        the quadratic of smooth_atom.

        The proximal function is (almost) a nullop.
        """
        smooth_atom = zero_smooth(proximal_atom.primal_shape)
        return simple_problem(smooth_atom, proximal_atom)

    def solve(self, quadratic=None, return_optimum=False, **fit_args):
        if quadratic is not None:
            oldq, self.quadratic = self.quadratic, self.quadratic + quadratic
        else:
            oldq = self.quadratic

        solver = FISTA(self)
        solver.composite.coefs[:] = self.coefs
        solver.fit(**fit_args)
        self.final_inv_step = solver.inv_step

        if return_optimum:
            value = (self.objective(self.coefs), self.coefs)
        else:
            value = self.coefs
        self.quadratic = oldq
        return value

    
def gengrad(simple_problem, L, tol=1.0e-8, max_its=1000, debug=False,
            coef_stop=False):
    """
    A simple generalized gradient solver
    """
    itercount = 0
    coef = simple_problem.coefs
    v = np.inf
    while True:
        vnew, g = simple_problem.smooth_objective(coef, 'both')
        vnew += simple_problem.nonsmooth_objective(coef)
        newcoef = simple_problem.proximal(identity_quadratic(L, coef, g, 0))
        if coef_stop:
            coef_stop_check = (np.linalg.norm(coef-newcoef) <= tol * 
                               np.max([np.linalg.norm(coef),
                                       np.linalg.norm(newcoef), 
                                       1]))
            if coef_stop_check:
                break
        else:
            obj_stop_check = np.fabs(v - vnew) <= tol * np.max([vnew, 1])
            if obj_stop_check:
                break
        if itercount == max_its:
            break
        if debug:
            print itercount, vnew, v, (vnew - v) / vnew
        v = vnew
        itercount += 1
        coef = newcoef
    return coef

def nesta(smooth_atom, proximal_atom, conjugate_atom, epsilon=None,
          tol=1.e-06):
    '''
    Parameters
    ==========

    smooth_atom: smooth_composite
        A smooth function, i.e. having a smooth_objective method.

    proximal_atom: 
        An atom with a proximal method.

    conjugate_atom:
        An atom that will be smoothed, by adding a quadratic to its
        conjugate.

    epsilon: np.array
        A decreasing array of positive constants for Moreau-Yosida smoothing.
        
    tol: np.float
        Tolerance to which each problem is solved is max(tol, epsilon)
    
    Returns
    =======

    primal: np.array
        Primal coefficients.

    dual: np.array
        Dual coefficients.

    '''
    if epsilon is None:
        epsilon = 2.**(-np.arange(20))

    transform, conjugate = conjugate_atom.dual
    dual_coef = np.zeros(conjugate_atom.dual_shape)
    for eps in epsilon:
        smoothed = conjugate_atom.smoothed(identity_quadratic(eps, dual_coef, 0, 0))
        if smooth_atom is not None:
            final_smooth = smooth_sum([smooth_atom, smoothed])
        else:
            final_smooth = smoothed
        problem = simple_problem(final_smooth, proximal_atom)
        primal_coef = problem.solve(tol=max(eps,tol))
        # when there's an affine transform involved
        dual_coef[:] = smoothed.grad
    
    return primal_coef, dual_coef

def tfocs(primal_atom, transform, dual_proximal_atom, epsilon=None,
          tol=1.e-06):
    '''

    This function is based on the setup of problems
    described in `TFOCS <http://tfocs.stanford.edu/>`_.
    Generally speaking, these are the same type of problems
    that nesta can handle, though without the additional smooth part.

    This solver is suited to solving problems of the form

    minimize_v f(v) + h(Dv+a)

    when both f and h (and hence f^* and h^*) 
    have simple proximal operators.

    Here is an example for minimum :math:`\ell_1` norm reconstruction.

    >>> import numpy as np, regreg.api as rr
    >>> n, p = 200, 5000

    The problem assumes Y=np.dot(X,beta) for some sparse beta.

    >>> X = np.random.standard_normal((n, p))
    >>> beta = np.zeros(p)
    >>> beta[:10] = 10
    >>> Y = np.dot(X, beta)
    >>> 

    The problem is formally,

    minimize_v np.fabs(v).sum() subject to Y=np.dot(X,v)

    The :math:`\ell_1` norm is described as:

    >>> l1 = rr.l1norm(p, lagrange=1)

    The constraint is specified as

    >>> constraint = rr.zero_constraint.affine(X,-Y)

    >>> transform, zero = constraint.dual
    >>> primal_tfocs, dual_tfocs = rr.tfocs(l1, transform, zero)
    >>> primal_tfocs[:20]
    array([ 10.00000055,  10.00000043,  10.00000038,  10.00000073,
             9.99999991,  10.00000058,   9.99999946,  10.00000004,
             9.99999981,  10.00000028,   0.        ,  -0.        ,
            -0.        ,   0.        ,   0.        ,  -0.        ,
             0.        ,   0.        ,  -0.        ,   0.        ])
    >>> np.linalg.norm(primal_tfocs[10:])
    0.0
    >>> 

    Parameters
    ==========

    primal_atom: atom
        An atom that will be smoothed,
        then composed with the transform.

    transform : affine_transform
        An affine transform for the composition.

    dual_proximal_atom: atom
        An atom with a proximal method.

    epsilon: np.array
        A decreasing array of positive constants for Moreau-Yosida smoothing.
        
    tol: np.float
        Tolerance to which each problem is solved is max(tol, epsilon)
    
    Returns
    =======

    primal: np.array
        Primal coefficients.

    dual: np.array
        Dual coefficients.

    
    '''
    transform = astransform(transform)

    #conjugate_atom needs a conjugate so it can be smoothed

    conjugate_atom = primal_atom.conjugate

    if epsilon is None:
        epsilon = 2.**(-np.arange(20))

    offset = transform.affine_offset
    if offset is not None:
        dual_sq = identity_quadratic(0,0,-offset, 0)
    else:
        dual_sq = identity_quadratic(0,0,0,0)
        
    primal_coef = np.zeros(conjugate_atom.primal_shape)
    for eps in epsilon:
        smoothed = conjugate_atom.smoothed(identity_quadratic(eps, primal_coef, 0, 0))
        final_smooth = affine_smooth(smoothed, scalar_multiply(adjoint(transform), -1))
        problem = simple_problem(final_smooth, dual_proximal_atom)
        dual_coef = problem.solve(dual_sq, tol=max(eps,tol))
        # when there's an affine transform involved
        primal_coef[:] = final_smooth.grad
    
    return primal_coef, dual_coef
