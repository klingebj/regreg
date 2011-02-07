
class linmodel(object):

    def __init__(self, data, **kwargs):
        
        self.penalties = self.default_penalty()
        if 'penalties' in kwargs:
            self.assign_penalty(**kwargs['penalties'])
        if 'initial_coefs' in kwargs:
            self.initial_coefs = kwargs['initial_coefs']
        if 'update_resids' in kwargs:
            self.update_resids = kwargs['update_resids']
        else:
            self.update_resids = True


        self.initialize(data, **kwargs)
    
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




    
class lasso(linmodel):

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

    def initialize(self, data, **kwargs):
        """
        Generate initial tuple of arguments for update.
        """
        
        if len(data) == 2:
            self.X = data[0]
            self.Y = data[1]
        else:
            raise ValueError("Data tuple not as expected")

        self._ssq = col_ssq(self.X)
        self.set_default_coefficients()
        if hasattr(self,'initial_coefs'):
            self.set_coefficients(self.initial_coefs)
        if 'L' in kwargs:
            self._L = kwargs['L']

    def default_penalty(self):
        """
        Default penalty for Lasso: a single
        parameter problem.
        """
        return np.zeros(1, np.dtype([(l, np.float) for l in ['l1']]))

    def set_default_coefficients(self):
        self.set_coefficients(np.zeros(len(self.X[0])))

    def obj(self,x):
        x = np.asarray(x)
        #return ((self.Y - np.dot(self.X,self.beta))**2).sum() / (2.*len(self.Y)) + np.sum(np.fabs(self.beta)) * self.penalties['l1']
        return ((self.Y - np.dot(self.X,x))**2).sum() / (2.*len(self.Y)) + np.sum(np.fabs(x)) * self.penalties['l1']

    def f(self,x):
        #return ((self.Y - np.dot(self.X,self.beta))**2).sum() / (2.*len(self.Y))
        x = np.asarray(x)
        return ((self.Y - np.dot(self.X,x))**2).sum() / (2.*len(self.Y))


    def gradf(self,x):
        #return (multlist(self.X,np.dot(self.X, self.beta),transpose=True) - np.dot(self.Y,self.X)) / (1.*len(self.Y))
        x = np.asarray(x)
        return (multlist(self.X,np.dot(self.X, x),transpose=True) - np.dot(self.Y,self.X)) / (1.*len(self.Y))

    def soft_thresh(self, x, g, L):
        v = x - g / L
        return np.sign(v) * np.maximum(np.fabs(v)-self.penalties['l1']/L, 0)

    def smooth(self, L, epsilon):
        return l1smooth.l1smooth(self.gradf, L, epsilon, l1=self.penalties['l1'], f=self.f)

    def _get_L(self):
        if hasattr(self,'_L'):
            return self._L
        else:
            #Power method
            v = np.random.normal(0,1,len(self.coefficients))
            change = np.inf
            norm_old = 0.
            while change > 0.01:            
                v = multlist(self.X,np.dot(self.X, v),transpose=True)
                norm = np.linalg.norm(v)
                change = np.fabs(norm-norm_old)
                norm_old = norm
                v /= norm
            return 1.01 * norm / (1.*len(self.Y))
            
    L = property(_get_L)


    def update_cwpath(self,
                      active,
                      list nonzero,
                      DTYPE_int_t inner_its = 1,
                      DTYPE_int_t permute = False,
                      DTYPE_int_t update_nonzero = False):
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
                                 inner_its,
                                 update_nonzero)

                
class lasso_wts(linmodel):

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
        self._ssq = col_ssq(self.X,self.rowweights)
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
                             self.rowweights,
                             inner_its)


class graphnet(linmodel):

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

    def initialize(self, data, **kwargs):
        """
        Generate initial tuple of arguments for update.
        """
        if len(data) == 3:
            self.X = data[0]
            self.Y = data[1]
            self.adj = data[2]
        else:
            raise ValueError("Data tuple not as expected")

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
        
    def update_cwpath(self,
                      active,
                      nonzero,
                      inner_its=1,
                      permute = False,
                      update_nonzero = False):
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
                                    inner_its,
                                    update_nonzero)

class lin_graphnet(linmodel):

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

    def initialize(self, data, **kwargs):
        """
        Generate initial tuple of arguments for update.
        """

        if len(data) == 3:
            self.X = data[0]
            self.Y = data[1]
            self.adj = data[2]
        elif len(data)==2:
            self.X = data[0]
            self.Y = data[1]
        else:
            raise ValueError("Data tuple not as expected")
        
        self.inner = col_inner(self.X,self.Y)
        self.set_default_coefficients()
        if hasattr(self,'initial_coefs'):
            self.set_coefficients(self.initial_coefs)
        if hasattr(self,'adj'):
            self.nadj = _create_nadj(self.adj)
        else:
            #Put in a placeholder for adj, nadj
            self.adj = np.zeros((2,2),dtype=int)
            self.nadj = np.zeros(2,dtype=int)
        if 'orth' in kwargs:
            self.orth = kwargs['orth']
        else:
            self.orth = np.zeros(self.coefficients.shape)

    def default_penalty(self):
        """
        Default penalty for Naive Laplace: a single
        parameter problem.
        """
        return np.zeros(1, np.dtype([(l, np.float) for l in ['l1', 'l2', 'l3','eta']]))

    def set_default_coefficients(self):
        self.set_coefficients(np.zeros(len(self.X[0])))
        
    def update_cwpath(self,
                      active,
                      nonzero,
                      inner_its=1,
                      permute = False,
                      update_nonzero = False):
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
                                        self.orth,
                                        inner_its,
                                        update_nonzero)



class v_graphnet(linmodel):

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
 
    name = "v_graphnet"

    def initialize(self, data, **kwargs):
        """
        Generate initial tuple of arguments for update.
        """

        if len(data) == 2:
            self.v = data[0]
            self.adj = data[1]
        else:
            raise ValueError("Data tuple not as expected")

        self.update_resids = False
        self.r = np.array([0.])
        self.set_default_coefficients()
        if hasattr(self,'initial_coefs'):
            self.set_coefficients(self.initial_coefs)
        if hasattr(self,'adj'):
            self.nadj = _create_nadj(self.adj)
        else:
            #Put in a placeholder for adj, nadj
            self.adj = np.zeros((2,2),dtype=int)
            self.nadj = np.zeros(2,dtype=int)



    def default_penalty(self):
        """
        Default penalty for Naive Laplace: a single
        parameter problem.
        """
        return np.zeros(1, np.dtype([(l, np.float) for l in ['l1', 'l2', 'l3']]))

    def set_default_coefficients(self):
        self.set_coefficients(np.zeros(len(self.v)))
        
    def update_cwpath(self,
                      active,
                      nonzero,
                      inner_its=1,
                      permute = False,
                      update_nonzero = False):
        """
        Update coefficients in active set, returning
        nonzero coefficients.
        """
        if permute:
            active = np.random.permutation(active)
        if len(active):
            _update_v_graphnet_cwpath(active,
                                        self.penalties,
                                        nonzero,
                                        self.beta,
                                        self.v,
                                        self.adj, 
                                        self.nadj,
                                        inner_its,
                                        update_nonzero)



class graphnet_wts(linmodel):

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

    def initialize(self, data, **kwargs):
        """
        Generate initial tuple of arguments for update.
        """
        if len(data) == 3:
            self.X = data[0]
            self.Y = data[1]
            self.adj = data[2]
        else:
            raise ValueError("Data tuple not as expected")


        
        self._ssq = col_ssq(self.X)
        self.penalty = self.default_penalty()
        self.nadj = _create_nadj(self.adj)
        self.set_default_coefficients()
        if hasattr(self,'initial_coefs'):
            self.set_coefficients(self.initial_coefs)
        if 'rowweights' in kwargs:
            self.set_rowweights(kwargs['rowweights'])
        else:
            raise ValueError("No weights given")

    def default_penalty(self):
        """
        Default penalty for Naive Laplace: a single
        parameter problem.
        """
        return np.zeros(1, np.dtype([(l, np.float) for l in ['l1', 'l2', 'l3']]))

    def set_default_coefficients(self):
        self.set_coefficients(np.zeros(len(self.X[0])))


            
    def update_cwpath(self,
                      active,
                      nonzero,
                      inner_its=1,
                      permute = False,
                      update_nonzero = False):
        """
        Update coefficients in active set, returning
        nonzero coefficients.
        """
        if permute:
            active = np.random.permutation(active)
        if len(active):
            _update_graphnet_wts_cwpath(active,
                                        self.penalties,
                                        nonzero,
                                        self.beta,
                                        self.r,
                                        self.X,
                                        self._ssq, 
                                        self.adj, 
                                        self.nadj,
                                        self.rowweights,
                                        inner_its,
                                        update_nonzero)

class univariate(linmodel):

    """
    LASSO problem with one penalty parameter
    Minimizes

    .. math::
       \begin{eqnarray}
       ||y - X\beta||^{2}_{2} + \lambda_{1}||\beta||_{1}
       \end{eqnarray}

    as a function of beta.
    """

    name = 'univariate'

    def initialize(self, data, **kwargs):
        """
        Generate initial tuple of arguments for update.
        """
        if len(data) == 2:
            self.X = data[0]
            self.Y = data[1]
        else:
            raise ValueError("Data tuple not as expected")
        self.inner = col_inner(self.X, self.Y)
        self.set_default_coefficients()

    def default_penalty(self):
        """
        There are no penalties for this problem
        """
        return np.zeros(1, np.dtype([(l, np.float) for l in ['l1']]))

    def set_default_coefficients(self):
        self.set_coefficients(np.zeros(len(self.X[0])))

    def update_direct(self):
        self.coefficients = soft_threshold(self.inner,self.penalties['l1'])
