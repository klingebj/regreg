# cython: profile=True

import numpy as np
cimport numpy as np
import time

## Local imports


## Compile-time datatypes
DTYPE_float = np.float
ctypedef np.float_t DTYPE_float_t
DTYPE_int = np.int
ctypedef np.int_t DTYPE_int_t

def regreg(data, problemtype, algorithm, **kwargs):
    #Create optimization algorithm
    #try:
    return algorithm(data, problemtype, **kwargs)
    #except:
    #    raise ValueError("Error creating algorithm class")

class cwpath(object):

    def __init__(self, data, problemtype, **kwargs):
        self.problem = problemtype(data, **kwargs)
        self.problem.initialize_cwpath(**kwargs)

            
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

    def fit(self,tol=1e-4,inner_its=25,max_its=200):
        
        cdef np.ndarray[DTYPE_int_t, ndim=1] all = np.arange(len(self.problem.beta))
        #cdef np.ndarray[DTYPE_float_t, ndim=1] bold
        cdef DTYPE_int_t stop = False
        cdef DTYPE_int_t count = 0
        cdef DTYPE_float_t worst
        while not stop and count < max_its:
            bold = self.copy()
            nonzero = []
            self.problem.update_cwpath(all,nonzero,1)
            stop, worst = self.stop(bold,tol=tol,return_worst=True)
         
            self.problem.update_cwpath(np.unique(nonzero),nonzero,inner_its)
            print count, worst
            count += 1
            #stop = self.stop(bold,tol=tol,return_worst=False)
            #print "Number active",  np.sum(np.fabs(bold[0])>0.)
            #print "Change", worst,  np.sum(np.fabs(bold[0])>0.), len(bold[0]), np.max(np.fabs(self.coefficients)), np.unique(bold[0]), np.max(np.fabs(self.response)), np.max(np.fabs(self.wts))
        
    def copy(self):
        """
        Copy relevant output.
        """
        cdef np.ndarray[DTYPE_float_t, ndim=1] coefs, r
        coefs, r = self.output()
        return (coefs.copy(), r.copy())
    

class regression(object):

    """
    def __init__(self, X, Y, adj=None, initial_coefs=None, weights=None, update_resids=True):
        
        self.images = X
        self.Y = Y.copy()
        self.adj = adj
        self.wts = weights
        self.update_resids = update_resids
        self.initial_coefs = initial_coefs
        self.initialize()
    """

    def __init__(self, data, **kwargs):

        if len(data) == 2:
            self.X = data[0]
            self.Y = data[1]
        elif len(data) == 3:
            self.X = data[0]
            self.Y = data[1]
            self.adj = data[2]
        else:
            raise ValueError("Data tuple not as expected")
        
        self.penalties = self.default_penalty()
        if 'penalties' in kwargs:
            self.assign_penalty(**kwargs['penalties'])
        if 'initial_coefs' in kwargs:
            self.initial_coefs = kwargs['initial_coefs']
        if 'update_resids' in kwargs:
            self.update_resids = kwargs['update_resids']
        else:
            self.update_resids = True
        if 'rowweights' in kwargs:
            self.set_rowweights(kwargs['rowweights'])

    
    def assign_penalty(self, **params):
        """
        Abstract method for assigning penalty parameters.
        """
        penalties = self.penalties.copy()
        for key in params:
            penalties[key] = params[key]
        self.penalties = penalties
        
    def set_coefficients(self, coefs):
        if coefs is not None:
            self.beta = coefs.copy()	
        if self.update_resids:    
            self.update_residuals()

    def get_coefficients(self):
        return self.beta.copy()

    coefficients = property(get_coefficients, set_coefficients)


    def set_response(self,Y):
        if Y is not None:
            self.Y = Y
            if self.update_resids:    
                self.update_residuals()
            if hasattr(self,'inner'):
                self.inner = col_inner(self.X,self.Y)

    def get_response(self):
        return self.Y.copy()

    response = property(get_response, set_response)

    def set_rowweights(self, weights):
        if weights is not None:
            self.rowwts = weights
            if self.update_resids:
                self.update_residuals()
            if hasattr(self,'_ssq'):
                self._ssq = col_ssq(self.X,self.rowwts)

    def get_rowweights(self):
        if hasattr(self,'rowwts'):
            return self.rowwts.copy()
        else:
            return None
    rowweights = property(get_rowweights, set_rowweights)


    def update_residuals(self):
        if hasattr(self,'rowwts'):
            self.r = self.Y - self.rowwts * np.dot(self.X,self.beta)    
        else:
            self.r = self.Y - np.dot(self.X,self.beta)

    def get_total_coefs(self):
        return self.beta.shape[0]
    total_coefs = property(get_total_coefs)




    
class lasso(regression):

    """
    LASSO problem with one penalty parameter
    Minimizes

    .. math::
       \begin{eqnarray}
       ||y - X\beta||^{2}_{2} + \lambda_{1}||\beta||_{1}
       \end{eqnarray}

    as a function of beta.
    """

    name = 'lasso'

    def initialize_cwpath(self, **kwargs):
        """
        Generate initial tuple of arguments for update.
        """
        self._ssq = col_ssq(self.X)
        self.set_default_coefficients()
        if hasattr(self,'initial_coefs'):
            self.set_coefficients(self.initial_coefs)

    def default_penalty(self):
        """
        Default penalty for Lasso: a single
        parameter problem.
        """
        return np.zeros(1, np.dtype([(l, np.float) for l in ['l1']]))

    def set_default_coefficients(self):
        self.set_coefficients(np.zeros(len(self.X[0])))

    def update_cwpath(self,
                      active,
                      list nonzero,
                      DTYPE_int_t inner_its = 1,
                      DTYPE_int_t permute = False):
        """
        Update coefficients in active set, returning
        nonzero coefficients.
        """
        if permute:
            active = np.random.permutation(active)
        if len(active):
            _update_lasso_cwpath(active,
                                 self.penalties,
                                 nonzero,
                                 self.beta,
                                 self.r,
                                 self.X,
                                 self._ssq,
                                 inner_its)

                
class lasso_wts(regression):

    """
    LASSO problem with one penalty parameter
    Minimizes

    .. math::
       \begin{eqnarray}
       ||y - X\beta||^{2}_{2} + \lambda_{1}||\beta||_{1}
       \end{eqnarray}

    as a function of beta.
    """

    name = 'lasso_wts'

    def initialize(self):
        """
        Generate initial tuple of arguments for update.
        """
        self._ssq = col_ssq(self.X,self.wts)
        self.penalty = self.default_penalty()
        self.set_coefficients(self.initial_coefs)



    def default_penalty(self):
        """
        Default penalty for Lasso: a single
        parameter problem.
        """
        #c = np.fabs(np.dot(self.X.T, self.Y)).max()
        return np.zeros(1, np.dtype([(l, np.float) for l in ['l1']]))
        

    def update(self, active, nonzero, inner_its=1, permute=False):
        """
        Update coefficients in active set, returning
        nonzero coefficients.
        """
        #print "nonzero", nonzero
        if permute:
            active = np.random.permutation(active)
        if len(active):
            _update_lasso_wts(active,
                             self.penalty,
                             nonzero,
                             self.beta,
                             self.r,
                             self.X,
                             self._ssq,
                             self.wts,
                             inner_its)


class graphnet(regression):

    """
    The Naive Laplace problem with three penalty parameters
    l1, l2 and l3 minimizes

    (np.sum((Y - np.dot(X, beta))**2) + l1 *
     np.fabs(beta).sum() + l2 * np.sum(beta**2))
     + l3* np.dot(np.dot(beta,np.dot(D-A)),beta)

     as a function of beta,
     where D = diag(N_1, ..., N_p) where N_i is the number
     of neighbors of coefficient i, and A_{ij} = 1(j is i's neighbor)

    """
 
    name = "graphnet"

    """
    def __init__(self, data, initial_coefs=None):
        
        if len(data) != 3:
            raise ValueError('expecting adjacency matrix for Laplacian smoothing')
        _, _, self.adj = data
        Regression.__init__(self, data[:2],initial_coefs)
    """

    def initialize_cwpath(self):
        """
        Generate initial tuple of arguments for update.
        """
        self._ssq = col_ssq(self.X)
        self.set_default_coefficients()
        if hasattr(self,'initial_coefs'):
            self.set_coefficients(self.initial_coefs)
        self.nadj = _create_nadj(self.adj)

    def default_penalty(self):
        """
        Default penalty for Naive Laplace: a single
        parameter problem.
        """
        return np.zeros(1, np.dtype([(l, np.float) for l in ['l1', 'l2', 'l3']]))

    def set_default_coefficients(self):
        self.set_coefficients(np.zeros(len(self.X[0])))
        
    def update_cwpath(self, active, nonzero, inner_its=1, permute=False):
        """
        Update coefficients in active set, returning
        nonzero coefficients.
        """
        if permute:
            active = np.random.permutation(active)
        if len(active):
            _update_graphnet_cwpath(active,
                                    self.penalties,
                                    nonzero,
                                    self.beta,
                                    self.r,
                                    self.X,
                                    self._ssq, 
                                    self.adj, 
                                    self.nadj,
                                    inner_its)

class lin_graphnet(regression):

    """
    The Naive Laplace problem with three penalty parameters
    l1, l2 and l3 minimizes

    (np.sum((Y - np.dot(X, beta))**2) + l1 *
     np.fabs(beta).sum() + l2 * np.sum(beta**2))
     + l3* np.dot(np.dot(beta,np.dot(D-A)),beta)

     as a function of beta,
     where D = diag(N_1, ..., N_p) where N_i is the number
     of neighbors of coefficient i, and A_{ij} = 1(j is i's neighbor)

    """
 
    name = "lin_graphnet"

    def initialize_cwpath(self):
        """
        Generate initial tuple of arguments for update.
        """
        
        self.inner = col_inner(self.X,self.Y)
        self.set_default_coefficients()
        if hasattr(self,'initial_coefs'):
            self.set_coefficients(self.initial_coefs)
        self.nadj = _create_nadj(self.adj)

    def default_penalty(self):
        """
        Default penalty for Naive Laplace: a single
        parameter problem.
        """
        return np.zeros(1, np.dtype([(l, np.float) for l in ['l1', 'l2', 'l3']]))

    def set_default_coefficients(self):
        self.set_coefficients(np.zeros(len(self.X[0])))
        
    def update_cwpath(self, active, nonzero, inner_its=1, permute=False):
        """
        Update coefficients in active set, returning
        nonzero coefficients.
        """
        if permute:
            active = np.random.permutation(active)
        if len(active):
            _update_lin_graphnet_cwpath(active,
                                        self.penalties,
                                        nonzero,
                                        self.beta,
                                        self.Y,
                                        self.X,
                                        self.inner, 
                                        self.adj, 
                                        self.nadj,
                                        inner_its)



class graphnet_wts(regression):

    """
    The Naive Laplace problem with three penalty parameters
    l1, l2 and l3 minimizes

    (np.sum((Y - np.dot(X, beta))**2) + l1 *
     np.fabs(beta).sum() + l2 * np.sum(beta**2))
     + l3* np.dot(np.dot(beta,np.dot(D-A)),beta)

     as a function of beta,
     where D = diag(N_1, ..., N_p) where N_i is the number
     of neighbors of coefficient i, and A_{ij} = 1(j is i's neighbor)

    """
 
    name = "graphnet_wts"

    """
    def __init__(self, data, initial_coefs=None):
        
        if len(data) != 3:
            raise ValueError('expecting adjacency matrix for Laplacian smoothing')
        _, _, self.adj = data
        Regression.__init__(self, data[:2],initial_coefs)
    """

    def initialize(self):
        """
        Generate initial tuple of arguments for update.
        """
        self._ssq = col_ssq(self.X)
        self.penalty = self.default_penalty()
        self.nadj = _create_nadj(self.adj)
        self.set_coefficients(self.initial_coefs)

    def default_penalty(self):
        """
        Default penalty for Naive Laplace: a single
        parameter problem.
        """
        return np.zeros(1, np.dtype([(l, np.float) for l in ['l1', 'l2', 'l3']]))
        
    def update(self, active, nonzero, inner_its=1, permute=False):
        """
        Update coefficients in active set, returning
        nonzero coefficients.
        """
        if permute:
            active = np.random.permutation(active)
        if len(active):
            _update_graphnet_wts(active,
                                 self.penalty,
                                 nonzero,
                                 self.beta,
                                 self.r,
                                 self.X,
                                 self._ssq, 
                                 self.adj, 
                                 self.nadj,
                                 self.wts,
                                 inner_its)



cdef _update_lasso_cwpath(np.ndarray[DTYPE_int_t, ndim=1] active,
                         penalty,
                         list nonzero,
                         np.ndarray[DTYPE_float_t, ndim=1] beta,
                         np.ndarray[DTYPE_float_t, ndim=1] r,
                         list X,
                         np.ndarray[DTYPE_float_t, ndim=1] ssq,
                         DTYPE_int_t inner_its,
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
                                DTYPE_int_t inner_its,
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
    n = len(Y)

    count = 0
    stop = False
    while count < inner_its and not stop:
        bold = beta.copy()
        for j in range(q):
            i = active[j]
            S = inner[i]
            lin, quad = _compute_Lbeta(adj,nadj,beta,i)
            new = _solve_plin(l3*quad/2. + l2/2.,
                              -S+l3*lin/2., 
                              l1)
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
            if new != 0:
                nonzero.append(i)
            db = beta[i] - new
            r += (db * col)
            beta[i] = new
        stop = coefficientCheck(bold,beta,tol)
        count += 1


def col_inner(list X,
              np.ndarray[DTYPE_float_t, ndim=1] Y):

    cdef DTYPE_int_t n, p, k, i
    cdef np.ndarray[DTYPE_float_t, ndim=1] inner = np.empty(p)
    n = len(Y)
    p = len(X[0])
    for i in range(p):
        inner[i] = 0.
        for k in range(n):
            inner[i] = inner[i] + Y[k]*X[k][i]
    return inner


def col_ssq(list X, wts = None):
    cdef DTYPE_int_t n, p, i, j
    n = len(X)
    p = len(X[0])
    cdef np.ndarray[DTYPE_float_t, ndim=1] ssq = np.zeros(p)
    if wts is not None:
        for i in range(p):
            for j in range(n):
                ssq[i] = ssq[i] + (X[j][i] * wts[j])**2
    else:
        for i in range(p):
            for j in range(n):
                ssq[i] = ssq[i] + (X[j][i])**2
    return ssq        


def _create_adj(DTYPE_int_t p):
    """
    Create default adjacency list, parameter i having neighbors
    i-1, i+1.
    """
    cdef list adj = []
    adj.append(np.array([p-1,1]))
    for i in range(1,p-1):
        adj.append(np.array([i-1,i+1]))
    adj.append(np.array([p-2,0]))
    return adj


def _create_nadj(np.ndarray[DTYPE_int_t, ndim=2] adj):
    """
    Create vector counting the number of neighbors each
    coefficient has.
    """
    cdef np.ndarray[DTYPE_int_t, ndim=1] nadj = np.zeros(adj.shape[0],dtype=DTYPE_int)
    for i in range(adj.shape[0]):
        nadj[i] = adj[i].shape[0]
    return nadj


def _compute_Lbeta(np.ndarray[DTYPE_int_t, ndim=2] adj,
                   np.ndarray[DTYPE_int_t, ndim=1] nadj,
                   np.ndarray[DTYPE_float_t, ndim=1] beta,
                   DTYPE_int_t k):
    """
    Compute the coefficients of beta[k] and beta[k]^2 in beta.T*2(D-A)*beta
    """

    cdef double quad_term = nadj[k]
    cdef double linear_term = 0
    cdef np.ndarray[DTYPE_int_t, ndim=1] row
    cdef int i
    
    row = adj[k]
    for i in range(row.shape[0]):
        linear_term += beta[row[i]]

    return -2*linear_term, quad_term



cdef DTYPE_int_t coefficientCheck(np.ndarray[DTYPE_float_t, ndim=1] bold,
                                  np.ndarray[DTYPE_float_t, ndim=1] bnew,
                                  DTYPE_float_t tol):

   #Check if all coefficients have relative errors < tol

   cdef DTYPE_int_t N = len(bold)
   cdef DTYPE_int_t i,j

   for i in range(N):
       if bold[i] == 0.:
           if bnew[i] != 0.:
               return False
       if np.fabs(np.fabs(bold[i]-bnew[i])/bold[i]) > tol:
           return False
   return True



cdef coefficientCheckVal(np.ndarray[DTYPE_float_t, ndim=1] bold,
                         np.ndarray[DTYPE_float_t, ndim=1] bnew,
                         DTYPE_float_t tol):
        
        #Check if all coefficients have relative errors < tol
        
        cdef long N = len(bold)
        cdef long i,j
        cdef DTYPE_float_t max_so_far = 0.
        cdef DTYPE_float_t max_active = 0.
        cdef DTYPE_float_t ratio = 0.

        for i in range(N):
            if bold[i] == 0.:
                if bnew[i] !=0.:
                    max_so_far = 10.
            else:
                ratio = np.fabs(np.fabs(bold[i]-bnew[i])/bold[i])
                if ratio > max_active:
                    max_active = ratio

        if max_active > max_so_far:
            max_so_far = max_active

        return max_so_far < tol, max_active

                   
cdef DTYPE_float_t _solve_plin(DTYPE_float_t a,
                               DTYPE_float_t b,
                               DTYPE_float_t c):
    """
    Find the minimizer of

    a*x**2 + b*x + c*fabs(x)

    for positive constants a, c and arbitrary b.
    """

    if b < 0:
        if b > -c:
            return 0.
        else:
            return -(c + b) / (2.*a)
    else:
        if c > b:
            return 0.
        else:
            return (c - b) / (2.*a)

