import numpy as np
from scipy import sparse

class affine_transform(object):
    
    def __init__(self, linear_operator, affine_offset, diag=False):
        self.affine_offset = affine_offset
        self.linear_operator = linear_operator

        if linear_operator is None:
            self.noneD = True
            self.sparseD = False
            self.affineD = False
            self.diagD = False
            self.primal_shape = affine_offset.shape
            self.dual_shape = affine_offset.shape
        else:
            self.noneD = False
            self.sparseD = sparse.isspmatrix(self.linear_operator)
            self.sparseD_csr = sparse.isspmatrix_csr(self.linear_operator)
            if self.sparseD_csr:
                self.linear_operator_T = sparse.csr_matrix(self.linear_operator.T)


            # does it support the affine_transform API
            if np.alltrue([hasattr(self.linear_operator, n) for 
                           n in ['linear_map',
                                 'affine_map',
                                 'affine_offset',
                                 'adjoint_map',
                                 'affine_objective',
                                 'primal_shape',
                                 'dual_shape']]):
                self.primal_shape = self.linear_operator.primal_shape
                self.dual_shape = self.linear_operator.dual_shape
                self.affineD = True
                self.diagD = False
            elif linear_operator.ndim == 1 and not diag:
                self.linear_operator = self.linear_operator.reshape((1,-1))
                self.diagD = False
                self.affineD = False
                self.primal_shape = (self.linear_operator.shape[1],)
                self.dual_shape = (1,)
            elif linear_operator.ndim == 1 and diag:
                self.diagD = True
                self.affineD = False
                self.primal_shape = (linear_operator.shape[0],)
                self.dual_shape = (linear_operator.shape[0],)
            else:
                self.primal_shape = (linear_operator.shape[1],)
                self.dual_shape = (linear_operator.shape[0],)
                self.diagD = False
                self.affineD = False

    def linear_map(self, x, copy=True):
        r"""
        Return :math:`Dx`

        This routine is subclassed in affine_atom
        as a matrix multiplications, but could
        also call FFTs if D is a DFT matrix, in a subclass.
        """

        if self.noneD:
            # this sometimes has to be a copy
            # because the array can later be modified
            # in place -- see the smoothed_seminorm
            if copy:
                return x.copy()
            return x
        elif self.affineD:
            return self.linear_operator.linear_map(x)
        else:
            if self.sparseD or self.diagD:
                return self.linear_operator * x
            else:
                return np.dot(self.linear_operator, x)

    def affine_map(self, x, copy=True):
        r"""
        Return :math:`Dx+\alpha`

        This routine is subclassed in affine_atom
        as a matrix multiplications, but could
        also call FFTs if D is a DFT matrix, in a subclass.
        """

        if self.affineD:
            v = self.linear_operator.affine_map(x)
        else:
            v = self.linear_map(x, copy)
        if self.affine_offset is not None:
            return self.affine_offset + v
        else:
            # if copy is True, v will already be a copy, so no need to check 
            # again
            return v

    def adjoint_map(self, u, copy=True):
        r"""
        Return :math:`D^Tu`

        This routine is currently a matrix multiplication, but could
        also call FFTs if D is a DFT matrix, in a subclass.
        """
        if not self.noneD:
            if self.sparseD or self.diagD:
                if self.sparseD_csr:
                    return self.linear_operator_T * u
                else:
                    return u * self.linear_operator
            elif self.affineD:
                return self.linear_operator.adjoint_map(u)
            else:
                return np.dot(u, self.linear_operator)
        else:
            # this might have to be a copy
            # but we only multiply by D.T when
            # computing gradient -- 
            # this currently doesn't happen in seminorm or
            # smoothed_seminorm
            if copy:
                return u.copy()
            else:
                return u

class linear_transform(affine_transform):

    def __init__(self, linear_operator, diag=False):
        affine_transform.__init__(self, linear_operator, None, diag=diag)

    def affine_map(self, u):
        raise ValueError('linear transforms have no affine part')

class selector(object):

    """
    Apply an affine transform after applying an
    indexing operation to the array.
    """
    def __init__(self, index_obj, initial_shape, affine_transform=None):
        self.index_obj = index_obj
        self.initial_shape = initial_shape

        if affine_transform == None:
            test = np.empty(initial_shape)
            affine_transform = identity(test[index_obj].shape)
        self.affine_transform = affine_transform
        self.affine_offset = self.affine_transform.affine_offset
        self.primal_shape = initial_shape
        self.dual_shape = self.affine_transform.dual_shape

    def linear_map(self, x, copy=True):
        x_indexed = x[self.index_obj]
        return self.affine_transform.linear_map(x_indexed)

    def affine_map(self, x, copy=True):
        x_indexed = x[self.index_obj]
        return self.affine_transform.affine_map(x_indexed)

    def adjoint_map(self, u, copy=True):
        if not hasattr(self, "_output"):
            self._output = np.zeros(self.initial_shape)
        self._output[self.index_obj] = self.affine_transform.adjoint_map(u)
        return self._output

class normalize(object):

    '''
    Normalize column by means and possibly scale. Could make
    a class for row normalization to.

    Columns are normalized to have std equal to value.
    '''

    def __init__(self, M, center=True, scale=True, value=1, inplace=False):
        '''
        Parameters
        ----------
        M : ndarray or scipy.sparse
            The matrix to be normalized. If an ndarray and inplace=True,
            then the values of M are modified in place. Sparse matrices
            are not modified in place.

        center : bool
            Center the columns?

        scale : bool
            Scale the columns?

        value : float
            Set the std of the columns to be value.

        inplace : bool
            If an ndarray and True, modify values in place.

        '''
        n, p = M.shape
        self.primal_shape = (p,)
        self.dual_shape = (n,)
        self.M = M

        self.sparseD = sparse.isspmatrix(self.M)

        self.center = center
        self.scale = scale

        if self.center:
            col_means = np.mean(M,0)
            if self.scale:
                self.invcol_scalings = np.sqrt((np.sum(M**2,0) - n * col_means**2) / n) * value
            if not self.sparseD and inplace:
                self.M -= col_means[np.newaxis,:]
                self.M /= self.invcol_scalings[np.newaxis,:]
        elif self.scale:
            self.invcol_scalings = np.sqrt(np.sum(M**2,0) / n) # or n-1?
            if not self.sparseD and inplace:
                self.M /= self.invcol_scalings[np.newaxis,:]
        self.affine_offset = None

    def linear_map(self, x):
        if self.scale:
            x = x / self.invcol_scalings
        if self.sparseD:
            v = self.M * x
        else:
            v = np.dot(self.M, x)
        if self.center:
            v -= v.mean()
        return v

    def affine_map(self, x):
        return self.linear_map(x)

    def adjoint_map(self, u):
        if self.center:
            u = u - u.mean()
        if self.sparseD:
            v = u * self.M
        else:
            v = np.dot(u, self.M)
        if self.scale:
            v /= self.invcol_scalings
        return v

class identity(object):

    def __init__(self, primal_shape):
        self.primal_shape = self.dual_shape = primal_shape
        self.affine_offset = None
        self.linear_operator = None

    def affine_map(self, x, copy=True):
        return self.linear_map(x, copy)

    def linear_map(self, x, copy=True):
        if copy:
            return x.copy()
        else:
            return x

    def adjoint_map(self, x, copy=True):
        return self.linear_map(x, copy)

