
class cwpath(object):

    def __init__(self, problem, **kwargs):
        self.problem = problem
            
    def stop(self,
             previous,
             DTYPE_float_t tol=1e-4,
             DTYPE_int_t return_worst = False):
        """
        Convergence check: check whether 
        residuals have not significantly changed or
        they are small enough.

        Both old and current are expected to be (beta, r) tuples, i.e.
        regression coefficent and residual tuples.
    
        """


        cdef np.ndarray[DTYPE_float_t, ndim=1] bold, bcurrent
        bold, _ = previous
        bcurrent, _ = self.output()

        if return_worst:
            status, worst = coefficientCheckVal(bold, bcurrent, tol)
            if status:
                return True, worst
            return False, worst
        else:
            status = coefficientCheck(bold, bcurrent, tol)
            if status:
                return True
            return False


    def output(self):
        """
        Return the 'interesting' part of the problem arguments.
        
        In the regression case, this is the tuple (beta, r).
        """
        return self.problem.coefficients, self.problem.r

    def fit(self,tol=1e-4,inner_its=50,max_its=2000,min_its=5):
        
        cdef np.ndarray[DTYPE_int_t, ndim=1] all = np.arange(len(self.problem.beta))
        #cdef np.ndarray[DTYPE_float_t, ndim=1] bold
        cdef DTYPE_int_t stop = False
        cdef DTYPE_int_t count = 0
        cdef DTYPE_float_t worst = np.inf
        while not stop and count < max_its:
            bold = self.copy()
            nonzero = []
            self.problem.update_cwpath(all,nonzero,1,update_nonzero=True)
            if count > min_its:
                stop, worst = self.stop(bold,tol=tol,return_worst=True)
                if np.mod(count,40)==0:
                    print "Fit iteration", count, "with max. relative change", worst
            self.problem.update_cwpath(np.unique(nonzero),nonzero,inner_its)
            count += 1

    def copy(self):
        """
        Copy relevant output.
        """
        cdef np.ndarray[DTYPE_float_t, ndim=1] coefs, r
        coefs, r = self.output()
        return (coefs.copy(), r.copy())


cdef _update_lasso_cwpath(np.ndarray[DTYPE_int_t, ndim=1] active,
                          penalty,
                          list nonzero,
                          np.ndarray[DTYPE_float_t, ndim=1] beta,
                          np.ndarray[DTYPE_float_t, ndim=1] r,
                          list X,
                          np.ndarray[DTYPE_float_t, ndim=1] ssq,
                          DTYPE_int_t inner_its,
                          DTYPE_int_t update_nonzero = False,
                          DTYPE_float_t tol = 1e-3):

                
    """
    Do coordinate-wise update of lasso coefficients beta
    in active set.

    The coordinates that are nonzero are stored in the
    list nonzero.

    Optimizes the LASSO penalty

    norm(Y-dot(X,b))**2/2 + penalty*fabs(b).sum()

    as a function of b.

    """
    cdef DTYPE_float_t S, lin, quad, new, db, l1
    cdef DTYPE_int_t q, n, i, j, k, m, count, stop
    cdef np.ndarray[DTYPE_float_t, ndim=1] bold

    l1 = float(penalty['l1'])
    q = active.shape[0]
    n = r.shape[0]
    cdef np.ndarray[DTYPE_float_t, ndim=1] col = np.empty(n)

    count = 0
    stop = False 
    while count < inner_its and not stop:
        bold = beta.copy()
        for j in range(q):
            i = active[j]
            
            #Select appropriate column
            #for k in range(n):
            #    col[k] = X[k][i]
            col = select_col(X,n,i)
                            
            S = beta[i] * ssq[i]
            S += np.dot(col,r)
            new = _solve_plin(ssq[i]/(2*n),
                              -(S/n),
                              l1)
            if update_nonzero:
                if new != 0:
                    nonzero.append(i)
            db = beta[i] - new
            r += (db * col)
            beta[i] = new
        stop = coefficientCheck(bold,beta,tol)
        count += 1

cdef select_col(list X,
                DTYPE_int_t n,
                DTYPE_int_t i):

    cdef np.ndarray[DTYPE_float_t, ndim=1] col = np.empty(n)
    cdef DTYPE_int_t k
    for k in range(n):
        col[k] = X[k][i]
    return col

def _update_lasso_wts(np.ndarray[DTYPE_int_t, ndim=1] active,
                      penalty,
                      list nonzero,
                      np.ndarray[DTYPE_float_t, ndim=1] beta,
                      np.ndarray[DTYPE_float_t, ndim=1] r,
                      list X,
                      np.ndarray[DTYPE_float_t, ndim=1] ssq,
                      np.ndarray[DTYPE_float_t, ndim=1] wts,
                      DTYPE_int_t inner_its,
                      DTYPE_int_t update_nonzero = False,
                      DTYPE_float_t tol = 1e-3):
                
    """
    Do coordinate-wise update of lasso coefficients beta
    in active set.

    The coordinates that are nonzero are stored in the
    list nonzero.

    Optimizes the LASSO penalty

    norm(Y-dot(X,b))**2/2 + penalty*fabs(b).sum()

    as a function of b.

    """
    cdef DTYPE_float_t S, lin, quad, new, db, l1
    cdef DTYPE_int_t q, n, i, j, k, m, count, stop
    cdef np.ndarray[DTYPE_float_t, ndim=1] bold


    l1 = float(penalty['l1'])
    q = active.shape[0]
    n = r.shape[0]
    cdef np.ndarray[DTYPE_float_t, ndim=1] col = np.empty(n)

    count = 0
    stop = False 
    while count < inner_its and not stop:
        bold = beta.copy()
        for j in range(q):
            i = active[j]
            
            #Select appropriate column
            for k in range(n):
                col[k] = X[k][i] * wts[k]
            
                            
            S = beta[i] * ssq[i] 
            S += np.dot(col,r)
            new = _solve_plin(ssq[i]/(2*n),
                              -(S/n),
                              l1)
            if update_nonzero:
                if new != 0:
                    nonzero.append(i)
            db = beta[i] - new
            r += (db * col)
            beta[i] = new
        stop = coefficientCheck(bold,beta,tol)
        count += 1
    #print count

def _update_graphnet_cwpath(np.ndarray[DTYPE_int_t, ndim=1] active,
                            penalty,
                            list nonzero,
                            np.ndarray[DTYPE_float_t, ndim=1] beta,
                            np.ndarray[DTYPE_float_t, ndim=1] r,
                            list X,
                            np.ndarray[DTYPE_float_t, ndim=1] ssq,
                            np.ndarray[DTYPE_int_t, ndim=2] adj,
                            np.ndarray[DTYPE_int_t, ndim=1] nadj,
                            DTYPE_int_t inner_its,
                            DTYPE_int_t update_nonzero = False,
                            DTYPE_float_t tol = 1e-3):

    """
    Do coordinate-wise update of lasso coefficients beta
    in active set.

    The coordinates that are nonzero are stored in the
    list nonzero.

    """
    cdef double S, lin, quad, new, db, l1, l2, l3
    cdef long q, n, i, j, k, count, stop
    cdef np.ndarray[DTYPE_float_t, ndim=1] bold


    l1 = float(penalty['l1'])
    l2 = float(penalty['l2'])
    l3 = float(penalty['l3'])
    q = active.shape[0]
    n = r.shape[0]
    cdef np.ndarray[DTYPE_float_t, ndim=1] col = np.empty(n)

    count = 0
    stop = False
    while count < inner_its and not stop:
        bold = beta.copy()
        for j in range(q):
            i = active[j]

            #Select appropriate column
            for k in range(n):
                col[k] = X[k][i]

                
            S = beta[i] * ssq[i]
            S += np.dot(col,r)

            lin, quad = _compute_Lbeta(adj,nadj,beta,i)
            new = _solve_plin(ssq[i]/(2*n) + l3*quad/2. + l2/2.,
                              -(S/n)+l3*lin/2., 
                              l1)
            if update_nonzero:
                if new != 0:
                    nonzero.append(i)
            db = beta[i] - new
            r += (db * col)
            beta[i] = new
        stop = coefficientCheck(bold,beta,tol)
        count += 1


def _update_lin_graphnet_cwpath(np.ndarray[DTYPE_int_t, ndim=1] active,
                                penalty,
                                list nonzero,
                                np.ndarray[DTYPE_float_t, ndim=1] beta,
                                np.ndarray[DTYPE_float_t, ndim=1] Y,
                                list X,
                                np.ndarray[DTYPE_float_t, ndim=1] inner,
                                np.ndarray[DTYPE_int_t, ndim=2] adj,
                                np.ndarray[DTYPE_int_t, ndim=1] nadj,
                                np.ndarray[DTYPE_float_t, ndim=1] orth,
                                DTYPE_int_t inner_its,
                                DTYPE_int_t update_nonzero = False,
                                DTYPE_float_t tol = 1e-3):

    """
    Do coordinate-wise update of lasso coefficients beta
    in active set.

    The coordinates that are nonzero are stored in the
    list nonzero.

    """
    cdef double S, lin, quad, new, db, l1, l2, l3
    cdef long q, n, i, j, k, count, stop
    cdef np.ndarray[DTYPE_float_t, ndim=1] bold


    l1 = float(penalty['l1'])
    l2 = float(penalty['l2'])
    l3 = float(penalty['l3'])
    eta = float(penalty['eta'])
    q = active.shape[0]
    n = len(Y)

    count = 0
    stop = False
    while count < inner_its and not stop:
        bold = beta.copy()
        for j in range(q):
            i = active[j]
            S = inner[i]
            if l3 > 0.:
                lin, quad = _compute_Lbeta(adj,nadj,beta,i)
            else:
                lin = 0.
                quad = 0.
            new = _solve_plin(l3*quad/2. + l2/2.,
                              -S+l3*lin/2. + eta*orth[i], 
                              l1)
            if update_nonzero:
                if new != 0:
                    nonzero.append(i)
            db = beta[i] - new
            beta[i] = new
        stop = coefficientCheck(bold,beta,tol)
        count += 1



def _update_v_graphnet_cwpath(np.ndarray[DTYPE_int_t, ndim=1] active,
                              penalty,
                              list nonzero,
                              np.ndarray[DTYPE_float_t, ndim=1] beta,
                              np.ndarray[DTYPE_float_t, ndim=1] v,
                              np.ndarray[DTYPE_int_t, ndim=2] adj,
                              np.ndarray[DTYPE_int_t, ndim=1] nadj,
                              DTYPE_int_t inner_its,
                              DTYPE_int_t update_nonzero = False,
                              DTYPE_float_t tol = 1e-3):

    """
    Do coordinate-wise update of lasso coefficients beta
    in active set.

    The coordinates that are nonzero are stored in the
    list nonzero.

    """
    cdef double S, lin, quad, new, db, l1, l2, l3
    cdef long q, n, i, j, k, count, stop
    cdef np.ndarray[DTYPE_float_t, ndim=1] bold


    l1 = float(penalty['l1'])
    l2 = float(penalty['l2'])
    l3 = float(penalty['l3'])
    q = active.shape[0]

    count = 0
    stop = False
    while count < inner_its and not stop:
        bold = beta.copy()
        for j in range(q):
            i = active[j]
            S = v[i]
            if l3 > 0.:
                lin, quad = _compute_Lbeta(adj,nadj,beta,i)
            else:
                lin = 0.
                quad = 0.
            new = _solve_plin(l3*quad/2. + l2/2.,
                              -S+l3*lin/2., 
                              l1)
            if update_nonzero:
                if new != 0:
                    nonzero.append(i)
            db = beta[i] - new
            beta[i] = new
        stop = coefficientCheck(bold,beta,tol)
        count += 1




def _update_graphnet_wts(np.ndarray[DTYPE_int_t, ndim=1] active,
                         penalty,
                         list nonzero,
                         np.ndarray[DTYPE_float_t, ndim=1] beta,
                         np.ndarray[DTYPE_float_t, ndim=1] r,
                         list X,
                         np.ndarray[DTYPE_float_t, ndim=1] ssq,
                         np.ndarray[DTYPE_int_t, ndim=2] adj,
                         np.ndarray[DTYPE_int_t, ndim=1] nadj,
                         np.ndarray[DTYPE_float_t, ndim=1] wts,
                         DTYPE_int_t inner_its,
                         DTYPE_int_t update_nonzero = False,
                         DTYPE_float_t tol = 1e-3):

    """
    Do coordinate-wise update of lasso coefficients beta
    in active set.

    The coordinates that are nonzero are stored in the
    list nonzero.

    """
    cdef double S, lin, quad, new, db, l1, l2, l3
    cdef long q, n, i, j, k, count, stop
    cdef np.ndarray[DTYPE_float_t, ndim=1] bold


    l1 = float(penalty['l1'])
    l2 = float(penalty['l2'])
    l3 = float(penalty['l3'])
    q = active.shape[0]
    n = r.shape[0]
    cdef np.ndarray[DTYPE_float_t, ndim=1] col = np.empty(n)

    count = 0
    stop = False
    while count < inner_its and not stop:
        bold = beta.copy()
        for j in range(q):
            i = active[j]

            #Select appropriate column
            for k in range(n):
                col[k] = X[k][i] * wts[k]

                
            S = beta[i] * ssq[i]
            S += np.dot(col,r)

            lin, quad = _compute_Lbeta(adj,nadj,beta,i)
            new = _solve_plin(ssq[i]/(2*n) + l3*quad/2. + l2/2.,
                              -(S/n)+l3*lin/2., 
                              l1)
            if update_nonzero:
                if new != 0:
                    nonzero.append(i)
            db = beta[i] - new
            r += (db * col)
            beta[i] = new
        stop = coefficientCheck(bold,beta,tol)
        count += 1


def _update_graphnet_wts_cwpath(np.ndarray[DTYPE_int_t, ndim=1] active,
                                penalty,
                                list nonzero,
                                np.ndarray[DTYPE_float_t, ndim=1] beta,
                                np.ndarray[DTYPE_float_t, ndim=1] r,
                                list X,
                                np.ndarray[DTYPE_float_t, ndim=1] ssq,
                                np.ndarray[DTYPE_int_t, ndim=2] adj,
                                np.ndarray[DTYPE_int_t, ndim=1] nadj,
                                np.ndarray[DTYPE_float_t, ndim=1] wts,
                                DTYPE_int_t inner_its,
                                DTYPE_int_t update_nonzero = False,
                                DTYPE_float_t tol = 1e-3):

    """
    Do coordinate-wise update of lasso coefficients beta
    in active set.

    The coordinates that are nonzero are stored in the
    list nonzero.

    """
    cdef double S, lin, quad, new, db, l1, l2, l3
    cdef long q, n, i, j, k, count, stop
    cdef np.ndarray[DTYPE_float_t, ndim=1] bold


    l1 = float(penalty['l1'])
    l2 = float(penalty['l2'])
    l3 = float(penalty['l3'])
    q = active.shape[0]
    n = r.shape[0]
    cdef np.ndarray[DTYPE_float_t, ndim=1] col = np.empty(n)

    count = 0
    stop = False
    while count < inner_its and not stop:
        bold = beta.copy()
        for j in range(q):
            i = active[j]

            #Select appropriate column
            for k in range(n):
                col[k] = X[k][i] * wts[k]

                
            S = beta[i] * ssq[i]
            S += np.dot(col,r)

            lin, quad = _compute_Lbeta(adj,nadj,beta,i)
            new = _solve_plin(ssq[i]/(2*n) + l3*quad/2. + l2/2.,
                              -(S/n)+l3*lin/2., 
                              l1)
            if update_nonzero:
                if new != 0:
                    nonzero.append(i)
            db = beta[i] - new
            r += (db * col)
            beta[i] = new
        stop = coefficientCheck(bold,beta,tol)
        count += 1
