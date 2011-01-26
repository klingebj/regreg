
class regression(object):

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
                                        inner_its,
                                        update_nonzero)



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


class univariate(regression):

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

    def initialize_direct(self, **kwargs):
        """
        Generate initial tuple of arguments for update.
        """
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
