import numpy as np
import warnings

class algorithm(object):

    def __init__(self, composite):
        self.composite = composite
        self.debug = False
        self.inv_step = None

    @property
    def output(self):
        """
        Return the 'interesting' part of the composite arguments.
        In the regression case, this is the tuple (beta, r).
        """
        return self.composite.output

    def fit(self):
        """
        Abstract method.
        """
        raise NotImplementedError

class FISTA(algorithm):

    """
    The FISTA generalized gradient algorithm
    """

    def fit(self,
            max_its=10000,
            min_its=5,
            tol=1e-5,
            backtrack=True,
            FISTA=True,
            alpha=1.1,
            start_inv_step=1.,
            restart=np.inf,
            coef_stop=False,
            return_objective_hist = True,
            monotonicity_restart=True,
            debug = None,
            prox_control=None,
            attempt_decrease = False):

        """
        Use the FISTA (or ISTA) algorithm to fit the problem

        Parameters
        ----------
        max_its : int
              the maximum number of iterations
        min_its : int
              the minimum number of iterations
        tol : float
              the tolerance used in the stopping criterion
        backtrack : bool
              use backtracking?
        FISTA : bool
              use Nesterov weights? If False, this is just gradient descent
        alpha : float
              used in backtracking. If the backtraking constant (self.inv_step) is too small, it is increased by a factor of alpha
        start_int_step : float
              used in backtracking. This is the starting value of self.inv_step
        restart : int
              Restart Nesterov weights every restart iterations. Default is never (np.inf)
        coef_stop : bool
              Stop based on coefficient changes instead of objective value
        return_objective_hist : bool
              Return the sequence of objective values?
        monotonicity_restart : bool
              If True, Nesterov weights are restarted every time the objective value increases
        debug : bool
              Resets self.debug, which controls whether convergence information is printed
        prox_control : dict
              A dictionary of arguments for fit(), used when the composite.proximal_step itself is a FISTA problem
        attempt_decrease : bool
              If True, attempt to decrease inv_step on the first iteration
    
        Returns
        -------

        objective_hist : ndarray
              A vector of objective values. Only return if return_objective_hist is True.

        """

        if debug is not None:
            self.debug = debug
        set_prox_control = prox_control is not None

        objective_hist = np.zeros(max_its)

        
        if backtrack and self.inv_step is None:
            #If inv_step is not available from last fit use start_inv_step
            self.inv_step = start_inv_step

        r = self.composite.coefs
        t_old = 1.

        beta = self.composite.coefs
        current_f = self.composite.smooth_objective(r,mode='func')
        current_obj = current_f + self.composite.nonsmooth_objective(self.composite.coefs, check_feasibility=True)
        
        itercount = 0
        badstep = 0
        while itercount < max_its:

            #Restart every 'restart' iterations
            if np.mod(itercount+1,restart)==0:
                if self.debug:
                    print "\tRestarting weights"
                r = self.composite.coefs
                t_old = 1.

            objective_hist[itercount] = current_obj

            # Backtracking loop
            if backtrack:
                if np.mod(itercount+1,100)==0 or attempt_decrease:
                    self.inv_step *= 1/alpha
                    attempt_decrease = True
                current_f, grad = self.composite.smooth_objective(r,mode='both')
                stop = False
                while not stop:
                    if set_prox_control:
                        beta = self.composite.proximal_step(self.inv_step, r, grad, prox_control=prox_control)
                    else:
                        beta = self.composite.proximal_step(self.inv_step, r, grad)

                    trial_f = self.composite.smooth_objective(beta,mode='func')

                    if not np.isfinite(trial_f):
                        stop = False
                    elif np.fabs(trial_f - current_f)/np.max([1.,trial_f]) > 1e-10:
                        stop = trial_f <= current_f + np.dot((beta-r).reshape(-1),grad.reshape(-1)) + 0.5*self.inv_step*np.linalg.norm(beta-r)**2
                    else:
                        trial_grad = self.composite.smooth_objective(beta,mode='grad')
                        stop = np.fabs(np.dot((beta-r).reshape(-1),(grad-trial_grad).reshape(-1))) <= 0.5*self.inv_step*np.linalg.norm(beta-r)**2
                    if not stop:
                        attempt_decrease = False
                        self.inv_step *= alpha
                        if not np.isfinite(self.inv_step):
                            raise ValueError("inv_step overflowed")
                        if self.debug:
                            print "%i    Increasing inv_step to" % itercount, self.inv_step
                     
            else:
                #Use specified Lipschitz constant
                grad = self.composite.smooth_objective(r,mode='grad')
                self.inv_step = self.composite.lipschitz
                if set_prox_control:
                    beta = self.composite.proximal_step(self.inv_step, r, grad, prox_control=prox_control)
                else:
                    beta = self.composite.proximal_step(self.inv_step, r, grad)
                trial_f = self.composite.smooth_objective(beta,mode='func')
                
            trial_obj = trial_f + self.composite.nonsmooth_objective(beta)

            obj_change = np.fabs(trial_obj - current_obj)
            #obj_rel_change = obj_change/np.fabs(max(min(current_obj, trial_obj),0))
            obj_rel_change = obj_change/np.max([np.fabs(current_obj),1.])
            if coef_stop:
                coef_rel_change = np.linalg.norm(self.composite.coefs - beta) / np.max([1.,np.linalg.norm(beta)])

            if self.debug:
                if coef_stop:
                    print itercount, current_obj, self.inv_step, obj_rel_change, coef_rel_change, tol
                else:
                    print "%i    obj: %.6e    inv_step: %.2e    rel_obj_change: %.2e    tol: %.1e" % (itercount, current_obj, self.inv_step, obj_rel_change, tol)

            if itercount >= min_its:
                if coef_stop:
                    if coef_rel_change < tol:
                        self.composite.coefs = beta
                        if self.debug:
                            print "Success: Optimization stopped because change in coefficients was below tolerance"
                        break
                else:
                    if obj_rel_change < tol or obj_change < tol:
                        self.composite.coefs = beta
                        if self.debug:
                            print 'Success: Optimization stopped because decrease in objective was below tolerance'
                        break

            if FISTA:
                #Use Nesterov weights
                t_new = 0.5 * (1 + np.sqrt(1+4*(t_old**2)))
                r = beta + ((t_old-1)/(t_new)) * (beta - self.composite.coefs)
            else:
                #Just do ISTA
                t_new = 1.
                r = beta

            if itercount > 1 and current_obj < trial_obj and obj_rel_change > 1e-10 and monotonicity_restart:
                #Adaptive restarting: restart if monotonicity violated
                if self.debug:
                    print "%i    Restarting weights" % itercount
                attempt_decrease = True

                if t_old == 1.:
                    #Gradient step didn't decrease objective: tolerance composites or incorrect prox op... time to give up?
                    if self.debug:
                        print "%i  Badstep: current: %f, proposed %f" % (itercount, current_obj, trial_obj)
                    badstep += 1
                    if badstep > 3:
                        warnings.warn('prox is taking bad steps')
                        if self.debug:
                            print 'Caution: Optimization stopped while prox was taking bad steps'
                        break
                itercount += 1
                t_old = 1.
                r = self.composite.coefs

            else:
                self.composite.coefs = beta
                t_old = t_new
                itercount += 1
                current_obj = trial_obj


        if self.debug:
            if itercount == max_its:
                print "Optimization stopped because iteration limit was reached"
            print "FISTA used", itercount, "of", max_its, "iterations"
        if return_objective_hist:
            return objective_hist[:itercount]


