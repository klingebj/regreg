"""
This module has utilities to take
a score function, add the smoothed
LASSO penalty and create a new callable
that evalutes this new function and its gradient.

This related to p.136, or (4.1) in the reference below.

References
----------

Nesterov, Y. "Smooth minimization of non-smooth functions."
Math. Program. 103 (2005), no. 1, Ser. A, 127--152. 

Candes, NESTA (TODO: add reference)

"""

import numpy as np

def l1smooth(gradf, M, epsilon, l1=1, Ainfo=None, f=None):
    """

    Take a tuple (f, gradf) and return the smoothed
    (f_epsilon, gradf_epsilon) as well as the new Lipschitz constant.
    We assume the prox function in the dual $d_2(x)=\|x\|^2_2$
    so $\sigma_2=1$.

    Parameters
    ----------

    gradf : callable
       This evaluates $\nabla f(x)$. It should take one argument of shape (p,).

    M : np.float
        This is a Lipschitz constant for the $\nabla f$.

    epsilon : np.float
        Smoothing parameter in (4.1).

    l1 : np.float
        Parameter that we will multiply A  by for the $\ell_1$ penalty.
        
    Ainfo : (np.float, np.float)
        A tuple (A, A_12) where A
        is the matrix corresponding to the $\ell_1$
        penalty we will use and A_12 is its operator norm.
        If not supplied, defaults to (np.identity(p), 1).

    f : callable
       This evaluates $f(x)$. It should take one argument of shape (p,).
       If present, then the smoothed version of $f$ is returned. 

    Returns
    -------

    gradf_epsilon : callable
        The gradient $\nabla f_{\epsilon}$, of the smoothed function
        with parameter epsilon.

    L_epsilon : np.float
        A Lipschitz constant of $\nabla f_{\epsilon}$.

    f_epsilon : callable
        If $f$ is supplied as an argument,
        the smoothed function $f_{\epsilon}$ is returned.

    """

    if Ainfo is None:

        # It is assumed here that Ainfo=(np.identity, 1)
        A_12 = 1

        def gradf_epsilon(x):
            u = u_eps(x, epsilon, l1)
            return gradf(x) + u
        
        # XXX this is
        # a little more work because
        # we compute t twice.
        # if the callables either returned 2-tuples or vectors
        # we could save computations.
        # This would require changing the loop function in the
        # nesterov module to recognize whether the callable
        # returns both or not.

        if f is not None:
            def f_epsilon(x):
                x = np.asarray(x)
                x = np.atleast_1d(x)
                u = u_eps(x, epsilon, l1)
                return f(x) + ((u*x).sum(0)-epsilon*(u**2).sum(0)/2)

    else:

        A, A_12 = Ainfo

        def gradf_epsilon(x):
            Ax = np.dot(A, x)
            u = u_eps(Ax, epsilon, l1)
            return gradf(x) + np.dot(A.T, u)
        
        # XXX this is
        # a little more work because
        # we compute t twice. see above
        
        if f is not None:
            def f_epsilon(x):
                Ax = np.dot(A, x)
                u = u_eps(Ax, epsilon, l1)
                return f(x) + ((x*np.dot(A.T, u)).sum(0) - epsilon*(u**2).sum(0)/2)

    L_epsilon = M + l1 * A_12**2 / epsilon

    if f is None:
        return gradf_epsilon, L_epsilon
    else:
        return gradf_epsilon, L_epsilon, f_epsilon

def u_eps(x, epsilon, l1):
    """
    epsilon-Huberized l1*\ell-1 norm.
    """
    x = np.asarray(x)
    shape = x.shape
    x = np.atleast_1d(x)
    t = np.less(np.fabs(x), l1*epsilon)
    u = np.empty(x.shape)
    u[t] = x[t] / epsilon
    u[~t] = l1*np.sign(x[~t])
    u.shape = x.shape
    return u

        
