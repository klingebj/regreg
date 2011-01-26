


def col_inner(list X,
              np.ndarray[DTYPE_float_t, ndim=1] Y):

    cdef DTYPE_int_t n, p, k, i
    p = len(X[0])
    cdef np.ndarray[DTYPE_float_t, ndim=1] inner = np.zeros(p)
    cdef np.ndarray[DTYPE_float_t, ndim=1] row
    n = len(Y)

    for k in range(n):
        row = X[k]
        for i in range(p):
            inner[i] = inner[i] + Y[k]*row[i]
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

"""
def _create_adj(DTYPE_int_t p):
    
    #Create default adjacency list, parameter i having neighbors
    #i-1, i+1.
    
    cdef list adj = []
    adj.append(np.array([p-1,1]))
    for i in range(1,p-1):
        adj.append(np.array([i-1,i+1]))
    adj.append(np.array([p-2,0]))
    return adj
"""

def _check_adj(np.ndarray[DTYPE_int_t, ndim=2] adj):
    """
    Check that the adjacenecy array is 'symmetric' and contains only valid indices
    """
    cdef DTYPE_int_t i, j, p, d, nbr
    p = adj.shape[0]
    d = adj.shape[1]

    for i in range(p):
        for j in range(d):
            nbr = adj[i,j]
            if nbr != -1:
                if not (nbr in range(p)):
                    raise ValueError("Adjacency array contains invalid indices")
                if not (i in adj[nbr]):
                    raise ValueError("Adjacency array is not symmetric")
    

def _create_nadj(np.ndarray[DTYPE_int_t, ndim=2] adj):
    """
    Create vector counting the number of neighbors each coefficient has.
    """
    cdef np.ndarray[DTYPE_int_t, ndim=1] nadj = np.zeros(adj.shape[0],dtype=DTYPE_int)
    cdef DTYPE_int_t i, j, p, d

    p = adj.shape[0]
    d = adj.shape[1]
    
    for i in range(p):
        for j in range(d):
            if adj[i,j] > -1:
                nadj[i] = nadj[i] + 1

    return nadj


def _compute_Lbeta(np.ndarray[DTYPE_int_t, ndim=2] adj,
                   np.ndarray[DTYPE_int_t, ndim=1] nadj,
                   np.ndarray[DTYPE_float_t, ndim=1] beta,
                   DTYPE_int_t k):
    """
    Compute the coefficients of beta[k] and beta[k]^2 in beta.T*2(D-A)*beta
    """


    cdef DTYPE_float_t quad_term = nadj[k]
    cdef DTYPE_float_t linear_term = 0
    cdef DTYPE_int_t j, p, d

    p = adj.shape[0]
    d = adj.shape[1]

    for j in range(d):
        if adj[k,j] > -1:
            linear_term += beta[adj[k,j]]

    return -2*linear_term, quad_term




def _mult_Lbeta(np.ndarray[DTYPE_int_t, ndim=2] adj,
                 np.ndarray[DTYPE_int_t, ndim=1] nadj,
                 np.ndarray[DTYPE_float_t, ndim=1] beta):
    """
    Compute \beta^T 2(D-A) \beta
    """

    cdef DTYPE_int_t i, j, p, d, nbr
    cdef DTYPE_float_t total = 0.
    cdef DTYPE_float_t subtotal 
    p = adj.shape[0]
    d = adj.shape[1]
    
    for i in range(p):
        subtotal = 0.
        for j in range(d):
            nbr = adj[i,j]
            if nbr > -1:
                subtotal = subtotal + beta[nbr]
        total = total + beta[i]*(beta[i]*nadj[i] - subtotal)

    return 2*total

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

def soft_threshold(np.ndarray[DTYPE_float_t, ndim=1] x,
                  DTYPE_float_t delta):

    """
    Applies the soft-thresholding operation to a vector x with parameter delta
    """

    cdef np.ndarray[DTYPE_float_t, ndim=1] signs = np.sign(x)
    cdef np.ndarray[DTYPE_float_t, ndim=1] v = np.fabs(x)-delta
    cdef DTYPE_int_t n = v.shape[0]
    cdef np.ndarray[DTYPE_float_t, ndim=1] pos = np.zeros(n)
    cdef DTYPE_int_t i
    for i in range(n):
        if v[i] > 0:
            pos[i] = 1
    return signs * v * pos
