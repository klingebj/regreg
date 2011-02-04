
class direct(object):

    def __init__(self, problem, **kwargs):
        self.problem = problem

    def output(self):
        """
        Return the 'interesting' part of the problem arguments.
        
        In the regression case, this is the tuple (beta, r).
        """
        return self.problem.coefficients, self.problem.r

    def fit(self,tol=1e-4):
        self.problem.update_direct()

    def copy(self):
        """
        Copy relevant output.
        """
        cdef np.ndarray[DTYPE_float_t, ndim=1] coefs, r
        coefs, r = self.output()
        return (coefs.copy(), r.copy())

