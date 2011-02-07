
class nesterov_eps(object):

    def __init__(self, problem, **kwargs):
        self.problem = problem

    def output(self):
        """                                                                                                                                                            Return the 'interesting' part of the problem arguments.                                                                                                        In the regression case, this is the tuple (beta, r).                                                                                                   
        """
        return self.problem.coefficients, self.problem.r

    def fit(self,tol=1e-4,epsilon=0.1,max_its=100):

        p = len(self.problem.coefficients)
        gradf_s, L_s, f_s = self.problem.smooth(self.problem.L, epsilon)
        self.problem.coefficients, l = nesterov_smooth.loop(self.problem.coefficients, gradf_s, L_s, f=f_s, maxiter=max_its, values=True, tol=tol)
        return f_s

    def copy(self):
        """                                                                                                                                                    
        Copy relevant output.                                                                                                                                  
        """
        cdef np.ndarray[DTYPE_float_t, ndim=1] coefs, r
        coefs, r = self.output()
        return (coefs.copy(), r.copy())
