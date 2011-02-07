
class fista(object):

    """
    Run FISTA algorithm
    """

    def __init__(self, problem, **kwargs):
        self.problem = problem

    def output(self):
        """                                                                                                                                                            Return the 'interesting' part of the problem arguments.                                                                                                        In the regression case, this is the tuple (beta, r).                                                                                                   
        """
        return self.problem.coefficients, self.problem.r

    def fit(self,max_its=100,tol=1e-5,miniter=5):

        f = self.problem.obj
        
        r = self.problem.coefficients
        t_old = 1.
        
        obj_cur = np.inf
        for itercount, param in enumerate(nesterov_smooth.parameters(max_its)):
            
            f_beta = f(self.problem.coefficients)
            if np.fabs((obj_cur - f_beta) / f_beta) < tol and itercount >= miniter:
                break
            obj_cur = f_beta
                    
            grad =  self.problem.gradf(r)
            beta = self.problem.soft_thresh(r, grad, self.problem.L)

            t_new = 0.5 * (1 + np.sqrt(1+4*(t_old**2)))
            r = beta + ((t_old-1)/(t_new)) * (beta - self.problem.coefficients)
            self.problem.coefficients = beta
            t_old = t_new

        print "FISTA used", itercount, "iterations"
    

    def copy(self):
        """                                                                                                                                                    
        Copy relevant output.                                                                                                                                  
        """
        cdef np.ndarray[DTYPE_float_t, ndim=1] coefs, r
        coefs, r = self.output()
        return (coefs.copy(), r.copy())
