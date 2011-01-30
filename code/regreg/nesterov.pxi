
class nesterov(object):

    def __init__(self, problem, **kwargs):
        self.problem = problem

    def output(self):
        """                                                                                                                                                            Return the 'interesting' part of the problem arguments.                                                                                                        In the regression case, this is the tuple (beta, r).                                                                                                   
        """
        return self.problem.coefficients, self.problem.r

    def fit(self,L,tol=1e-4,max_its=100):

        iter = 0
        obj_old = np.inf
        while iter < max_its:
            grad = self.problem.grad()
            self.problem.beta = self.problem.soft_thresh(self.problem.beta,grad,L)
            obj = self.problem.obj()
            if iter > 1:
                if np.fabs((obj-obj_old))/(obj_old) < tol:
                    break
            obj_old = obj
            iter += 1

    def copy(self):
        """                                                                                                                                                    
        Copy relevant output.                                                                                                                                  
        """
        cdef np.ndarray[DTYPE_float_t, ndim=1] coefs, r
        coefs, r = self.output()
        return (coefs.copy(), r.copy())
