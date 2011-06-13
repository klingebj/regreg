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

class ISTA(algorithm):

    def fit(self,
            max_its=10000,
            min_its=5,
            tol=1e-5,
            backtrack=True,
            alpha=1.1,
            start_inv_step=1.,
            coef_stop=False,
            prox_tol = None,
            prox_max_its = None,
            prox_debug = None,
            prox_lipshitz = None,
            prox_backtrack = None,
            prox_min_its = None):



        #Specify convergence criteria for proximal composite
        # This is a bit inconsistent: simple prox functions don't accept tolerance parameters, but when the prox function
        # is an optimization (like primal_prox) then it accepts some control paramters. This checks whether the user
        # gave the parameters before passing them on
        if (prox_tol is not None) or (prox_max_its is not None) or (prox_debug is not None) or (prox_lipshitz is not None) or (prox_backtrack is not None) or (prox_min_its is not None):
            set_prox_control = True
            if prox_tol is None:
                prox_tol = 1e-14
            if prox_max_its is None:
                prox_max_its = 5000
            if prox_min_its is None:
                prox_min_its = 5
            if prox_debug is None:
                prox_debug=False
            if prox_backtrack is None:
                prox_backtrack=False
            prox_control = {'tol':prox_tol,
                            'max_its':prox_max_its,
                            'debug':prox_debug}
        else:
            set_prox_control = False

        objective_hist = np.zeros(max_its)
        
        if self.inv_step is None:
            #If available, use Lipschitz constant from last fit
            self.inv_step = start_inv_step
        else:
            self.inv_step *= 1/alpha

        current_f = self.composite.smooth_objective(self.composite.coefs,mode='func')
        current_obj = current_f + self.composite.nonsmooth_objective(self.composite.coefs, check_feasibility=True)
        
        itercount = 0
        while itercount < max_its:

            objective_hist[itercount] = current_obj

            # Backtracking loop
            if backtrack:
                if np.mod(itercount+1,100)==0:
                    self.inv_step *= 1/alpha
                current_f, grad = self.composite.smooth_objective(self.composite.coefs,mode='both')
                stop = False
                while not stop:
                    if set_prox_control:
                        beta = self.composite.proximal_step(self.composite.coefs, grad, self.inv_step, prox_control=prox_control)
                    else:
                        beta = self.composite.proximal_step(self.composite.coefs, grad, self.inv_step)

                    trial_f = self.composite.smooth_objective(beta,mode='func')

                    if np.fabs(trial_f - current_f)/np.max([1.,trial_f]) > 1e-10:
                        stop = trial_f <= current_f + np.dot(beta-self.composite.coefs,grad) + 0.5*self.inv_step*np.linalg.norm(beta-self.composite.coefs)**2
                    else:
                        trial_grad = self.composite.smooth_objective(beta,mode='grad')
                        stop = np.fabs(np.dot(beta-self.composite.coefs,grad-trial_grad)) <= 0.5*self.inv_step*np.linalg.norm(beta-self.composite.coefs)**2
                    if not stop:
                        self.inv_step *= alpha
                        if self.debug:
                            print "Increasing inv_step", self.inv_step
                     
            else:
                #Use specified Lipschitz constant
                grad = self.composite.smooth_objective(self.composite.coefs,mode='grad')
                self.inv_step = self.composite.lipshitz
                if set_prox_control:
                    beta = self.composite.proximal_step(self.composite.coefs, grad, self.inv_step, prox_control=prox_control)
                else:
                    beta = self.composite.proximal_step(self.composite.coefs, grad, self.inv_step)
                trial_f = self.composite.smooth_objective(beta,mode='func')
                
            trial_obj = trial_f + self.composite.nonsmooth_objective(beta)

            obj_change = np.fabs(trial_obj - current_obj)
            obj_rel_change = obj_change/np.fabs(max(min(current_obj, trial_obj),0))

            if self.debug:
                print itercount, current_obj, self.inv_step, obj_rel_change, np.linalg.norm(self.composite.coefs - beta) / np.max([1.,np.linalg.norm(beta)]), tol


            if itercount >= min_its:
                if coef_stop:
                    if np.linalg.norm(self.composite.coefs - beta) / np.max([1.,np.linalg.norm(beta)]) < tol:
                        self.composite.coefs = beta
                        break
                else:
                    if obj_rel_change < tol or obj_change < tol:
                        self.composite.coefs = beta
                        break

            self.composite.coefs = beta
            itercount += 1
            current_obj = trial_obj
            
        if self.debug:
            print "ISTA used", itercount, "iterations"
        return objective_hist[:itercount]


class FISTA(algorithm):

    """
    The FISTA generalized gradient algorithm
    """

    def fit(self,
            max_its=10000,
            min_its=5,
            tol=1e-5,
            FISTA=True,
            backtrack=True,
            alpha=1.1,
            start_inv_step=1.,
            restart=np.inf,
            coef_stop=False,
            return_objective_hist = True,
            prox_tol = None,
            prox_max_its = None,
            prox_debug = None,
            monotonicity_restart=True,
            prox_lipshitz = None,
            prox_backtrack = None,
            prox_min_its = None):


        #Specify convergence criteria for proximal problem
        # This is a bit inconsistent: simple prox functions don't accept tolerance parameters, but when the prox function
        # is an optimization (like primal_prox) then it accepts some control paramters. This checks whether the user
        # gave the parameters before passing them on
        if (prox_tol is not None) or (prox_max_its is not None) or (prox_debug is not None) or (prox_lipshitz is not None) or (prox_backtrack is not None) or (prox_min_its is not None):
            set_prox_control = True
            if prox_tol is None:
                prox_tol = 1e-14
            if prox_max_its is None:
                prox_max_its = 5000
            if prox_min_its is None:
                prox_min_its = 5
            if prox_debug is None:
                prox_debug=False
            if prox_backtrack is None:
                prox_backtrack=False
            prox_control = {'tol':prox_tol,
                            'max_its':prox_max_its,
                            'debug':prox_debug}
        else:
            set_prox_control = False

        objective_hist = np.zeros(max_its)
        
        if self.inv_step is None:
            #If available, use Lipschitz constant from last fit
            self.inv_step = start_inv_step
        else:
            self.inv_step *= 1/alpha

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
                    print "Restarting"
                r = self.composite.coefs
                t_old = 1.

            objective_hist[itercount] = current_obj

            # Backtracking loop
            if backtrack:
                if np.mod(itercount+1,100)==0:
                    self.inv_step *= 1/alpha
                current_f, grad = self.composite.smooth_objective(r,mode='both')
                stop = False
                while not stop:
                    if set_prox_control:
                        beta = self.composite.proximal_step(r, grad, self.inv_step, prox_control=prox_control)
                    else:
                        beta = self.composite.proximal_step(r, grad, self.inv_step)

                    trial_f = self.composite.smooth_objective(beta,mode='func')

                    if np.fabs(trial_f - current_f)/np.max([1.,trial_f]) > 1e-10:
                        stop = trial_f <= current_f + np.dot(beta-r,grad) + 0.5*self.inv_step*np.linalg.norm(beta-r)**2
                    else:
                        trial_grad = self.composite.smooth_objective(beta,mode='grad')
                        stop = np.fabs(np.dot(beta-r,grad-trial_grad)) <= 0.5*self.inv_step*np.linalg.norm(beta-r)**2
                    if not stop:
                        self.inv_step *= alpha
                        if self.debug:
                            print "Increasing inv_step", self.inv_step
                     
            else:
                #Use specified Lipschitz constant
                grad = self.composite.smooth_objective(r,mode='grad')
                self.inv_step = self.composite.lipshitz
                if set_prox_control:
                    beta = self.composite.proximal_step(r, grad, self.inv_step, prox_control=prox_control)
                else:
                    beta = self.composite.proximal_step(r, grad, self.inv_step)
                trial_f = self.composite.smooth_objective(beta,mode='func')
                
            trial_obj = trial_f + self.composite.nonsmooth_objective(beta)

            obj_change = np.fabs(trial_obj - current_obj)
            obj_rel_change = obj_change/np.fabs(max(min(current_obj, trial_obj),0))

            if self.debug:
                print itercount, current_obj, self.inv_step, obj_rel_change, np.linalg.norm(self.composite.coefs - beta) / np.max([1.,np.linalg.norm(beta)]), tol

            if itercount >= min_its:

                if coef_stop:
                    if np.linalg.norm(self.composite.coefs - beta) / np.max([1.,np.linalg.norm(beta)]) < tol:
                        self.composite.coefs = beta
                        if self.debug:
                            print "Stopped because coefficients haven't changed"
                        break
                else:
                    if obj_rel_change < tol or obj_change < tol:
                        self.composite.coefs = beta
                        if self.debug:
                            print 'Stopped because objective value decreased'
                        break

            if FISTA:
                #Use Nesterov weights
                t_new = 0.5 * (1 + np.sqrt(1+4*(t_old**2)))
                r = beta + ((t_old-1)/(t_new)) * (beta - self.composite.coefs)
            else:
                #Just do ISTA
                t_new = 1.
                r = beta

            if current_obj < trial_obj and obj_rel_change > 1e-12 and current_obj > 1e-12 and monotonicity_restart:
                #Adaptive restarting: restart if monotonicity violated
                if self.debug:
                    print "\tRestarting", current_obj, trial_obj
                current_f = self.composite.smooth_objective(self.composite.coefs,mode='func')
                current_obj = current_f + self.composite.nonsmooth_objective(self.composite.coefs, check_feasibility=True)

                if not set_prox_control and t_old == 1.:
                    #Gradient step didn't decrease objective: tolerance composites or incorrect prox op... time to give up?
                    if self.debug:
                        print "Badstep: current: %f, proposed %f" % (current_obj, trial_obj)
                    badstep += 1
                    if badstep > 3:
                        warnings.warn('prox is taking bad steps')
                        if self.debug:
                            print 'Stopped in badstep'
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
            print "FISTA used", itercount, "iterations"
        if return_objective_hist:
            return objective_hist[:itercount]


