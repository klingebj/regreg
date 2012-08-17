import numpy as np, regreg.api as rr

class beta_loglikelihood_mu(rr.smooth_atom):
    
    def __init__(self, proportions, X,
                 coef=1.,
                 offset=None,
                 quadratic=None,
                 initial=None):
        X = ra.astransform(X)
        self.proportions = proportions.reshape(-1)
        self.primal_shape = (X.primal_shape[0]+1,)

        self.X = X
        rr.smooth_atom.__init__(self, self.primal_shape,
                                offset=offset,
                                coef=coef,
                                quadratic=quadratic,
                                initial=initial)
        self.ystar = np.log(proportions / (1. - proportions))
                             
    def smooth_objective(self, params, mode='both', check_feasibility=False):
        """
        Evaluate a smooth function and/or its gradient

        if mode == 'both', return both function value and gradient
        if mode == 'grad', return only the gradient
        if mode == 'func', return only the function value
        """
        phi = params[0]
        beta = params[1:]
        eta = self.X.linear_map(beta)
        exp_eta = np.exp(eta)
        mu = exp_eta / (1. + exp_eta)
        omega = mu * phi
        tau = (1 - mu) * phi
        dg_tau = scipy.special.digamma(tau)
        dg_omega = scipy.special.digamma(omega)
        mu_star = dg_omega - dg_tau
        
        W = mu * (1 - mu)
        if mode == 'func':
            v = scipy.special.betaln(omega, tau).sum()
            v -= ((omega - 1.) * np.log(self.proportions)).sum()
            v -= ((tau - 1.) * np.log(1.-self.proportions)).sum()
            return v
        elif mode == 'both':
            v = scipy.special.betaln(omega, tau).sum()
            v -= ((omega - 1.) * np.log(self.proportions)).sum()
            v -= ((tau - 1.) * np.log(1.-self.proportions)).sum()
            g = np.zeros(self.primal_shape)
            g[0]= -(mu * (self.ystar - mu_star) + np.log(1 - self.proportions) - dg_tau + scipy.special.digamma(phi)).sum()
            g[1:] = -phi * self.X.adjoint_map(W * (self.ystar - mu_star))
            return v, g
        elif mode == 'grad':
            g = np.zeros(self.primal_shape)
            g[0]= -(mu * (self.ystar - mu_star) + np.log(1 - self.proportions) - dg_tau + scipy.special.digamma(phi)).sum()
            g[1:] = -phi * self.X.adjoint_map(W * (self.ystar - mu_star))
            return g
        else:
            raise ValueError("mode incorrectly specified")
            
def georges_function(X, Y, nstep=100, tol=1.e-6, start_inv_step=2000., lagrange_proportion=.001):
    '''
    Parameters
    ==========

    X: ndarray
         Feature matrix with first column an intercept column

    Y: ndarray
         Proportions

    nstep: int
         How many points in the path?

    tol: float
         How accurately do we solve each problem?

    start_inv_step: float
         A guess at the Lipschitz constant of the loss function -- backtrack
         will find its own but this is an initial guess

    lagrange_proportion: 0 < float < 1
         What proportion of lagrange_max do we solve down to?

    '''
    # assuming that X has an intercept column as its first column
    
    # first, find lambda_max
    
    n, p = X.shape
    
    # loss function
    
    loss = beta_loglikelihood_mu(Y, X)
    
    null_design = X[:,0:1]
    null_problem = beta_loglikelihood_mu(Y,null_design)
    null_problem.coefs = np.ones(2)
    null_soln = null_problem.solve(tol=tol, start_inv_step=2000)

    full_null_soln = np.zeros(p+1)
    full_null_soln[:2] = null_soln
    grad_null = loss.smooth_objective(full_null_soln, mode='grad')
    lagrange_max = np.fabs(grad_null).max()

    # construct the penalty

    penalty = rr.group_lasso([rr.UNPENALIZED]*2+[rr.L1_PENALTY]*(p-1), lagrange_max)
    
    # and the problem
    
    problem = rr.simple_problem(loss, penalty)
    problem.coefs = np.ones(loss.primal_shape)

    solutions = []
    nstep = 100
    lagrange_proportion = 1.e-4
    lagrange_sequence = lagrange_max * np.exp(np.linspace(np.log(lagrange_proportion), 0,
                                    nstep))[::-1]
    for lagrange in lagrange_sequence:
        penalty.lagrange = lagrange
        soln = problem.solve(tol=1.e-6, start_inv_step=2000)
        solutions.append(soln.copy())
        
    return np.array(solutions)


