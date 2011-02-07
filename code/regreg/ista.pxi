
class ista(object):

    """
    Run the ISTA algorithm
    """


    def __init__(self, problem, **kwargs):
        self.problem = problem

    def output(self):
        """                                                                                                                                                            Return the 'interesting' part of the problem arguments.                                                                                                        In the regression case, this is the tuple (beta, r).                                                                                                   
        """
        return self.problem.coefficients, self.problem.r

    def fit(self,tol=1e-4,max_its=100,min_its=5):

        iter = 0
        obj_old = np.inf
        while iter < max_its:
            grad = self.problem.gradf(self.problem.coefficients)
            self.problem.beta = self.problem.soft_thresh(self.problem.coefficients,grad,self.problem.L)
            obj = self.problem.obj(self.problem.coefficients)
            if iter > min_its:
                if np.fabs((obj-obj_old))/(obj_old) < tol:
                    break
            obj_old = obj
            iter += 1
        print "ISTA used", iter, "iterations"

    def copy(self):
        """                                                                                                                                                    
        Copy relevant output.                                                                                                                                  
        """
        cdef np.ndarray[DTYPE_float_t, ndim=1] coefs, r
        coefs, r = self.output()
        return (coefs.copy(), r.copy())
