"""

Reference 
---------

Nesterov, Y. "Smooth minimization of non-smooth functions."
Math. Program. 103 (2005), no. 1, Ser. A, 127--152. 
"""

import numpy as np

def parameters(maxiter=100):
    """
    These are the $\alpha_k, \tau_k, A_k$ of (3.10)
    """
 
    for k in range(maxiter):
        alpha = (k+1)/2.
        tau = 2./(k+3)
        A = (k+1)*(k+2)/4.
        yield alpha, tau, A
#    return (k+1)/2., 2./(k+3), (k+1)*(k+2)/4.

def naive_solver(Q, b):
    """
    A solver that returns b if Q=None, else np.linalg.solve(Q,b).
    """
    if Q is None:
        return b
    else:
        return np.linalg.solve(Q, b)

def solve_z(momentum, L, Qinfo=None, solver=naive_solver,
            prox_center=None):
    """
    This solves for $z_k$ on p.133. We assume the
    prox function here is $d(x)=x'Qx/2$.

    Parameters
    ----------

    momentum : np.float
        This is the expression $\sum_i \alpha_i \nabla f(x_i)$ in
        the reference. It has shape (p,)

    L : np.float
        Lipschitz constant of $\nabla f$, the gradient of
        the function we are trying to minimize. That is, $\nabla f$ should 
        satisfy

        $$
        \| \nabla f(x) - \nabla f(y)\| \leq L \|x-y\|
        $$

    Qinfo : (np.float, np.float)
        A tuple (Q, sigma) where Q is the matrix in the
        prox-function (ridge penalty) and sigma is its
        smallest eigenvalue. Defaults to (np.identity(p), 1).

    solver : callable
        A function that will solve a system of the form
        $Qa=b$. It should take two arguments: (Q,b).

    prox_center : np.float
        A center for the prox function in the primal space.
        Should have shape (p,). Defaults to 0.

    Returns
    -------

    z : np.float
    """
    if Qinfo is None:
        sigma = 1.
        if solver != naive_solver:
            Q = np.identity(momentum.shape[0])
        else:
            Q = None
    if prox_center is not None:
        return solver(Q, prox_center + (-float(sigma)/L) * momentum)
    else:
        return solver(Q, (-float(sigma)/L) * momentum)
        

def solve_y(gradf_x, x, L):
    """
    This solves for $y_k$ on p.132.

    Parameters
    ----------

    gradf_x : np.float
       This is $\nabla f(x)$. It has shape (p,).

    x : np.float
       This has shape (p,)

    L : np.float
        Lipschitz constant of $\nabla f$, the gradient of
        the function we are trying to minimize. 
       
    Returns
    -------

    y : np.float
        This is the solution  to (3.2) on p.132. It is just
        x - gradf_x / L
    """
    return x - gradf_x / L

def loop(x0, gradf, L, Qinfo=None, maxiter=100, solver=naive_solver,
         f=None, tol=1.0e-12, values=False,
         miniter=4, prox_center=None):
    """
    This is the loop (3.11) on p.135.

    Parameters
    ----------

    x0 : np.float
       Starting point. It has shape (p,). 

    gradf : callable
       This evaluates $\nabla f(x)$. It should take one argument of shape (p,).

    L : np.float
        Lipschitz constant of $\nabla f$, the gradient of
        the function we are trying to minimize. 

    Qinfo : (np.float, np.float)
        A tuple (Q, sigma) where Q is the matrix in the
        prox-function (ridge penalty) and sigma is its
        smallest eigenvalue. Defaults to (np.identity(p), 1).

    sigma : np.float
        Convexity parameter of $d(x)=x'Qx/2$, which we can take to be
        the smallest eigenvalue of $Q$.

    solver : callable
        A function that will solve a system of the form
        $Qa=b$. It should take two arguments: (Q,b).

    f : callable
        The function we are trying to minimize. If
        present, then the value of the objective can be
        returned, and (3.14) of the reference can be used
        in order to have a monotone descent of $f(y_k)$.
        
    tol : float
        If f is not None, used to determine convergence
        based on the objective function. 

    values : bool
        If True, and f is not None return the values of $f(y_k)$.

    miniter : int
        Minimum number of iterations to run.

    prox_center : np.float
        A center for the prox function in the primal space.
        Should have shape (p,). Defaults to 0.

    Returns
    -------

    x : np.float
        This is the solution to (3.11) after maxiter steps.
    """

    x = x0; y=x0; y_old = y.copy()
    momentum = 0

    obj_cur = np.inf
    fvalues = []
    count = 0
    for itercount, param in enumerate(parameters(maxiter)):
        alpha, tau, A = param
        if f is not None:
            f_x = f(x)
            f_y = f(y)

            if np.fabs((obj_cur - f_x) / f_x) < tol and itercount >= miniter:
            #if np.fabs((obj_cur - f_y) / f_y) < tol and itercount >= miniter:
            #if np.allclose(y,y_old) and itercount >=miniter:
                count += 1
                if count > 10:
                    break
            else:
                count = 0
            obj_cur = f_y
            obj_cur = f_x
            #y_old = y.copy() 
            if values:
                fvalues.append(f_y)
        gradf_x =  gradf(x) 
        if f is None:
            y = solve_y(gradf_x, x, L)
        else:
            y_prime = solve_y(gradf_x, x, L)
            f_y_prime = f(y_prime)
            fs = [f_y, f_x, f_y_prime]
            idx = np.argmin(fs)
            y = [y, x, y_prime][idx]

        momentum += alpha * gradf_x
        z = solve_z(momentum, L, Qinfo=Qinfo, solver=solver,
                    prox_center=prox_center)
        x = tau * z + (1 - tau) * y

    print "Smoothed used", itercount, "iterations"

    if not values:
        return y
    else:
        return y, np.array(fvalues)

