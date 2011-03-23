import numpy as np

class algorithm(object):

    def __init__(self, problem):
        self.problem = problem
        self.debug = False
        self.inv_step = None

    @property
    def output(self):
        """
        Return the 'interesting' part of the problem arguments.
        In the regression case, this is the tuple (beta, r).
        """
        return self.problem.output

    def fit(self):
        """
        Abstract method.
        """
        raise NotImplementedError

class ISTA(algorithm):

    def fit(self,tol=1e-4,max_its=100,min_its=5,backtrack=True,alpha=1.1,start_inv_step=1.):

        objective_hist = np.zeros(max_its)
        itercount = 0
        obj_cur = np.inf
        if self.inv_step is None:
            self.inv_step = start_inv_step
        while itercount < max_its:
            f_beta = self.problem.obj(self.problem.coefs)
            grad = self.problem.grad(self.problem.coefs)
            objective_hist[itercount] = f_beta
            # Backtracking loop
            if backtrack:
                current_f = self.problem.obj_smooth(self.problem.coefs)
                stop = False
                while not stop:
                    beta = self.problem.proximal(self.problem.coefs, grad, self.inv_step)
                    trial_f = self.problem.obj_smooth(beta)
                    if np.fabs(trial_f - current_f)/np.max([1.,trial_f]) > 1e-10:
                        stop = trial_f <= current_f + np.dot(beta-self.problem.coefs,grad) + 0.5*self.inv_step*np.linalg.norm(beta-self.problem.coefs)**2
                    else:
                        trial_grad = self.problem.grad(beta)
                        stop = np.fabs(np.dot(beta-self.problem.coefs,grad-trial_grad)) <= 0.5*self.inv_step*np.linalg.norm(beta-self.problem.coefs)**2
                    if not stop:
                        self.inv_step *= alpha
            else:
                self.inv_step = self.problem.L
                beta = self.problem.proximal(self.problem.coefs, grad, self.inv_step)

            if self.debug:
                print itercount, obj_cur, self.inv_step, (obj_cur - f_beta) / f_beta, np.linalg.norm(self.problem.coefs - beta) / np.max([1.,np.linalg.norm(beta)])


            if np.linalg.norm(self.problem.coefs - beta) / np.max([1.,np.linalg.norm(beta)]) < tol and itercount >= min_its:
            #if np.fabs((obj_cur - f_beta) / f_beta) < tol and itercount >= min_its:
                self.problem.coefs = beta
                break
            self.problem.coefs = beta
            obj_cur = self.problem.obj(self.problem.coefs)
            

            itercount += 1
        if self.debug:
            print "ISTA used", itercount, "iterations"
        return objective_hist

class FISTA(algorithm):

    def fit(self,
            max_its=100,
            min_its=5,
            tol=1e-5,
            backtrack=True,
            alpha=1.1,
            start_inv_step=1.,
            restart=np.inf,
            coef_stop=False):

        objective_hist = np.zeros(max_its)
        itercount = 0
        if self.inv_step is None:
            #If available, use Lipschitz constant from last fit
            self.inv_step = start_inv_step
        r = self.problem.coefs
        t_old = 1.
        beta = self.problem.coefs
        current_f = self.problem.obj_smooth(r)
        current_obj = self.problem.obj(self.problem.coefs)
        while itercount < max_its:

            #Restart every restart iterations
            if np.mod(itercount+1,restart)==0:
                if self.debug:
                    print "Restarting"
                r = self.problem.coefs
                t_old = 1.
                self.inv_step *= 0.5            

            objective_hist[itercount] = current_obj
            grad = self.problem.grad(r)

            # Backtracking loop
            if backtrack:
                current_f = self.problem.obj_smooth(r)
                stop = False
                while not stop:
                    beta = self.problem.proximal(r, grad, self.inv_step)
                    trial_f = self.problem.obj_smooth(beta)

                    if np.fabs(trial_f - current_f)/np.max([1.,trial_f]) > 1e-10:
                        stop = trial_f <= current_f + np.dot(beta-r,grad) + 0.5*self.inv_step*np.linalg.norm(beta-r)**2
                    else:
                        trial_grad = self.problem.grad(beta)
                        stop = np.fabs(np.dot(beta-r,grad-trial_grad)) <= 0.5*self.inv_step*np.linalg.norm(beta-r)**2
                    if not stop:
                        self.inv_step *= alpha
                        #if self.debug:
                        #    print self.inv_step
            else:
                #Use specified Lipschitz constant
                self.inv_step = self.problem.L
                beta = self.problem.proximal(r, grad, self.inv_step)
                trial_f = self.problem.obj_smooth(beta)
                
            trial_obj = trial_f + self.problem.obj_rough(beta)
            obj_change = np.fabs(trial_obj - current_obj)
            obj_rel_change = obj_change/current_obj 

            if self.debug:
                print itercount, current_obj, self.inv_step, obj_rel_change, np.linalg.norm(self.problem.coefs - beta) / np.max([1.,np.linalg.norm(beta)]), tol


            if itercount >= min_its:
                if not coef_stop:
                    if obj_rel_change < tol:
                        self.problem.coefs = beta
                        break
                else:
                    if np.linalg.norm(self.problem.coefs - beta) / np.max([1.,np.linalg.norm(beta)]) < tol:
                        self.problem.coefs = beta
                        break
            else:
                if not coef_stop:
                    if obj_change < 0.:
                        self.problem.coefs = beta
                        break                
                    


            t_new = 0.5 * (1 + np.sqrt(1+4*(t_old**2)))
            r = beta + ((t_old-1)/(t_new)) * (beta - self.problem.coefs)
            
            if current_obj < trial_obj and obj_rel_change > 1e-10:
                #Adaptive restarting: restart if monotonicity violated
                if self.debug:
                    print "Restarting", trial_obj, current_obj
                t_old = 1.
                r = self.problem.coefs
            else:
                self.problem.coefs = beta
                t_old = t_new
                itercount += 1
                current_obj = trial_obj
            

        if self.debug:
            print "FISTA used", itercount, "iterations"
        return objective_hist[:itercount]


