import numpy as np

class Regression(object):

    def __init__(self, problem):
        self.problem = problem

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

    def copy(self):
        """
        Copy relevant output.
        """
        coefs, r = self.output
        return (coefs.copy(), r.copy())

class ISTA(Regression):

    def fit(self,L,tol=1e-4,max_its=100,min_its=5):

        itercount = 0
        obj_old = np.inf
        while itercount < max_its:
            grad = self.problem.grad(self.problem.coefficients)
            self.problem.beta = self.problem.proximal(self.problem.coefficients,grad,L)
            obj = self.problem.obj(self.problem.coefficients)
            if iter > min_its:
                if np.fabs((obj-obj_old))/(obj_old) < tol:
                    break
            obj_old = obj
            itercount += 1
        print "ISTA used", itercount, "iterations"

class FISTA(Regression):

    def fit(self,L,max_its=100,tol=1e-5,miniter=5):

        f = self.problem.obj
        
        r = self.problem.coefficients
        t_old = 1.
        
        obj_cur = np.inf
        itercount = 0
        while itercount < max_its:
            f_beta = f(self.problem.coefficients)
            if np.fabs((obj_cur - f_beta) / f_beta) < tol and itercount >= miniter:
                break
            obj_cur = f_beta
                    
            grad =  self.problem.grad(r)
            beta = self.problem.proximal(r, grad, L)

            t_new = 0.5 * (1 + np.sqrt(1+4*(t_old**2)))
            r = beta + ((t_old-1)/(t_new)) * (beta - self.problem.coefficients)
            self.problem.coefficients = beta
            t_old = t_new
            itercount += 1

        print "FISTA used", itercount, "iterations"
    
class NesterovSmooth(Regression):
    
    def fit(self,L,tol=1e-4,epsilon=0.1,max_its=100):
        import nesterov_smooth
        p = len(self.problem.coefficients)
        grad_s, L_s, f_s = self.problem.smooth(L, epsilon)
        self.problem.coefficients, l = nesterov_smooth.loop(self.problem.coefficients, grad_s, L_s, f=f_s, maxiter=max_its, values=True, tol=tol)
        return f_s


import subfunctions as sf
class CWPath(Regression):


    def __init__(self, problem, **kwargs):
        self.problem = problem
            
    def fit(self,tol=1e-4,inner_its=50,max_its=2000,min_its=5):
        
        active = np.arange(self.problem.beta.shape[0])
        itercount = 0
        stop = False
        while not stop and itercount < max_its:
            bold = self.copy()
            nonzero = []
            self.problem.update_cwpath(active,nonzero,1,update_nonzero=True)
            if itercount > min_its:
                stop, worst = self.stop(bold,tol=tol,return_worst=True)
                if np.mod(itercount,40)==0:
                    print "Fit iteration", itercount, "with max. relative change", worst
            self.problem.update_cwpath(np.unique(nonzero),nonzero,inner_its)
            itercount += 1

    def stop(self,
             previous,
             tol=1e-4,
             return_worst = False):
        """
        Convergence check: check whether 
        residuals have not significantly changed or
        they are small enough.

        Both old and current are expected to be (beta, r) tuples, i.e.
        regression coefficent and residual tuples.
    
        """

        bold, _ = previous
        bcurrent, _ = self.output

        if return_worst:
            status, worst = sf.coefficientCheckVal(bold, bcurrent, tol)
            if status:
                return True, worst
            return False, worst
        else:
            status = sf.coefficientCheck(bold, bcurrent, tol)
            if status:
                return True
            return False

class Direct(Regression):
    def fit(self,tol=1e-4):
        self.problem.update_direct()
