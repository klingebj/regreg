import numpy as np
cimport numpy as np
import time

## Local imports

#from regression import Regression
## Compile-time datatypes
DTYPE_float = np.float
ctypedef np.float_t DTYPE_float_t

DTYPE_int = np.int
ctypedef np.int_t DTYPE_int_t

class regression(object):

    def __init__(self, images, Y, adj=None, initial_coefs=None, weights=None, update_resids=True):
        
        self.images = images
        self.Y = Y.copy()
        self.adj = adj
        self.wts = weights
        self.update_resids = update_resids
        self.initial_coefs = initial_coefs
        self.initialize()


    def coefficientCheckVal(self,
                         np.ndarray[DTYPE_float_t, ndim=1] bold,
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
            status, worst = self.coefficientCheckVal(bold, bcurrent, tol)
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
        return self.coefficients, self.r

    def fit(self,tol=1e-4,max_its=5):
        
        all = np.arange(len(self.beta))
        stop = False
        while not stop:
            bold = self.copy()
            nonzero = []
            self.update(all,nonzero,1)
            stop, worst = self.stop(bold,tol=tol,return_worst=True)
            self.update(np.unique(nonzero),nonzero,max_its)
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

    
    def assign_penalty(self, path_key=None, **params):
        """
        Abstract method for assigning penalty parameters.
        """

        if path_key is None:
            path_length = 1
        else:
            path_length = len(params[path_key])
        penalty_list = []
        for i in range(path_length):
            penalty = self.penalty.copy()
            for key in params:
                if key==path_key:
                    penalty[key] = params[key][i]
                else:
                    penalty[key] = params[key]
            penalty_list.append(penalty)
        if path_length == 1:
            penalty_list = penalty_list[0]
        self.penalty = penalty_list    

        
    def set_coefficients(self, coefs):
        if coefs is not None:
            self.beta = coefs.copy()	
        else:
            self.beta = np.zeros(len(self.images[0]))
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

    def get_response(self):
        return self.Y.copy()

    response = property(get_response, set_response)

    def set_weights(self, weights):
        if weights is not None:
            self.wts = weights
            if self.update_resids:
                self.update_residuals()
            self._Xssq = col_ssq(self.images,self.wts)

    def get_weights(self):
        return self.wts.copy()

    weights = property(get_weights, set_weights)


    def update_residuals(self):
        if self.wts is not None:
            self.r = self.Y - self.wts * np.dot(self.images,self.beta)    
        else:
            self.r = self.Y - np.dot(self.images,self.beta)    

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

    def initialize(self):
        """
        Generate initial tuple of arguments for update.
        """
        self._Xssq = col_ssq(self.images)
        self.penalty = self.default_penalty()
        self.set_coefficients(self.initial_coefs)



    def default_penalty(self):
        """
        Default penalty for Lasso: a single
        parameter problem.
        """
        #c = np.fabs(np.dot(self.X.T, self.Y)).max()
        return np.zeros(1, np.dtype([(l, np.float) for l in ['l1']]))
        

    def update(self, active, nonzero, max_its=1, permute=False):
        """
        Update coefficients in active set, returning
        nonzero coefficients.
        """
        #print "nonzero", nonzero
        if permute:
            active = np.random.permutation(active)
        if len(active):
            _update_lasso(active,
                          self.penalty,
                          nonzero,
                          self.beta,
                          self.r,
                          self.images,
                          self._Xssq,
                          max_its)

                
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
        self._Xssq = col_ssq(self.images,self.wts)
        self.penalty = self.default_penalty()
        self.set_coefficients(self.initial_coefs)



    def default_penalty(self):
        """
        Default penalty for Lasso: a single
        parameter problem.
        """
        #c = np.fabs(np.dot(self.X.T, self.Y)).max()
        return np.zeros(1, np.dtype([(l, np.float) for l in ['l1']]))
        

    def update(self, active, nonzero, max_its=1, permute=False):
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
                             self.images,
                             self._Xssq,
                             self.wts,
                             max_its)



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

    def initialize(self):
        """
        Generate initial tuple of arguments for update.
        """
        self._Xssq = col_ssq(self.images)
        self.penalty = self.default_penalty()
        self.nadj = _create_nadj(self.adj)
        self.set_coefficients(self.initial_coefs)

    def default_penalty(self):
        """
        Default penalty for Naive Laplace: a single
        parameter problem.
        """
        return np.zeros(1, np.dtype([(l, np.float) for l in ['l1', 'l2', 'l3']]))
        
    def update(self, active, nonzero, max_its=1, permute=False):
        """
        Update coefficients in active set, returning
        nonzero coefficients.
        """
        if permute:
            active = np.random.permutation(active)
        if len(active):
            _update_graphnet(active,
                             self.penalty,
                             nonzero,
                             self.beta,
                             self.r,
                             self.images,
                             self._Xssq, 
                             self.adj, 
                             self.nadj,
                             max_its)


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
        self._Xssq = col_ssq(self.images)
        self.penalty = self.default_penalty()
        self.nadj = _create_nadj(self.adj)
        self.set_coefficients(self.initial_coefs)

    def default_penalty(self):
        """
        Default penalty for Naive Laplace: a single
        parameter problem.
        """
        return np.zeros(1, np.dtype([(l, np.float) for l in ['l1', 'l2', 'l3']]))
        
    def update(self, active, nonzero, max_its=1, permute=False):
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
                                 self.images,
                                 self._Xssq, 
                                 self.adj, 
                                 self.nadj,
                                 self.wts,
                                 max_its)



def _update_lasso(np.ndarray[DTYPE_int_t, ndim=1] active,
                  penalty,
                  list nonzero,
                  np.ndarray[DTYPE_float_t, ndim=1] beta,
                  np.ndarray[DTYPE_float_t, ndim=1] r,
                  list images,
                  np.ndarray[DTYPE_float_t, ndim=1] Xssq,
                  DTYPE_int_t max_its,
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
    while count < max_its and not stop:
        bold = beta.copy()
        for j in range(q):
            i = active[j]
            
            #Select appropriate column
            for k in range(n):
                col[k] = images[k][i]
            
                            
            S = beta[i] * Xssq[i]
            S += np.dot(col,r)
            new = _solve_plin(Xssq[i]/(2*n),
                              -(S/n),
                              l1)
            if new != 0:
                nonzero.append(i)
            db = beta[i] - new
            r += (db * col)
            beta[i] = new
        stop = coefficientCheck(bold,beta,tol)
        count += 1



def _update_lasso_wts(np.ndarray[DTYPE_int_t, ndim=1] active,
                      penalty,
                      list nonzero,
                      np.ndarray[DTYPE_float_t, ndim=1] beta,
                      np.ndarray[DTYPE_float_t, ndim=1] r,
                      list images,
                      np.ndarray[DTYPE_float_t, ndim=1] Xssq,
                      np.ndarray[DTYPE_float_t, ndim=1] wts,
                      DTYPE_int_t max_its,
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
    while count < max_its and not stop:
        bold = beta.copy()
        for j in range(q):
            i = active[j]
            
            #Select appropriate column
            for k in range(n):
                col[k] = images[k][i] * wts[k]
            
                            
            S = beta[i] * Xssq[i] 
            S += np.dot(col,r)
            new = _solve_plin(Xssq[i]/(2*n),
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


def _update_graphnet(np.ndarray[DTYPE_int_t, ndim=1] active,
                     penalty,
                     list nonzero,
                     np.ndarray[DTYPE_float_t, ndim=1] beta,
                     np.ndarray[DTYPE_float_t, ndim=1] r,
                     list images,
                     np.ndarray[DTYPE_float_t, ndim=1] Xssq,
                     np.ndarray[DTYPE_int_t, ndim=2] adj,
                     np.ndarray[DTYPE_int_t, ndim=1] nadj,
                     DTYPE_int_t max_its,
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
    while count < max_its and not stop:
        bold = beta.copy()
        for j in range(q):
            i = active[j]

            #Select appropriate column
            for k in range(n):
                col[k] = images[k][i]

                
            S = beta[i] * Xssq[i]
            S += np.dot(col,r)

            lin, quad = _compute_Lbeta(adj,nadj,beta,i)
            new = _solve_plin(Xssq[i]/(2*n) + l3*quad/2. + l2/2.,
                              -(S/n)+l3*lin/2., 
                              l1)
            if new != 0:
                nonzero.append(i)
            db = beta[i] - new
            r += (db * col)
            beta[i] = new
        stop = coefficientCheck(bold,beta,tol)
        count += 1



def _update_graphnet_wts(np.ndarray[DTYPE_int_t, ndim=1] active,
                         penalty,
                         list nonzero,
                         np.ndarray[DTYPE_float_t, ndim=1] beta,
                         np.ndarray[DTYPE_float_t, ndim=1] r,
                         list images,
                         np.ndarray[DTYPE_float_t, ndim=1] Xssq,
                         np.ndarray[DTYPE_int_t, ndim=2] adj,
                         np.ndarray[DTYPE_int_t, ndim=1] nadj,
                         np.ndarray[DTYPE_float_t, ndim=1] wts,
                         DTYPE_int_t max_its,
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
    while count < max_its and not stop:
        bold = beta.copy()
        for j in range(q):
            i = active[j]

            #Select appropriate column
            for k in range(n):
                col[k] = images[k][i] * wts[k]

                
            S = beta[i] * Xssq[i]
            S += np.dot(col,r)

            lin, quad = _compute_Lbeta(adj,nadj,beta,i)
            new = _solve_plin(Xssq[i]/(2*n) + l3*quad/2. + l2/2.,
                              -(S/n)+l3*lin/2., 
                              l1)
            if new != 0:
                nonzero.append(i)
            db = beta[i] - new
            r += (db * col)
            beta[i] = new
        stop = coefficientCheck(bold,beta,tol)
        count += 1



def col_ssq(list images, wts = None):
    cdef DTYPE_int_t n, p, i, j
    n = len(images)
    p = len(images[0])
    cdef np.ndarray[DTYPE_float_t, ndim=1] ssq = np.zeros(p)
    if wts is not None:
        for i in range(p):
            for j in range(n):
                ssq[i] = ssq[i] + (images[j][i] * wts[j])**2
    else:
        for i in range(p):
            for j in range(n):
                ssq[i] = ssq[i] + (images[j][i])**2
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

   cdef long N = len(bold)
   cdef long i,j

   for i in range(N):
       if bold[i] == 0.:
           if bnew[i] != 0.:
               return False
       if np.fabs(np.fabs(bold[i]-bnew[i])/bold[i]) > tol:
           return False
   return True

                   
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

