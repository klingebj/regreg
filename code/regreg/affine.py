from operator import add, mul
import numpy as np
from scipy import sparse


def broadcast_first(a, b, op):
    """ apply binary operation `op`, broadcast `a` over axis 1 if necessary

    Parameters
    ----------
    a : ndarray
        If a is 1D shape (N,), convert to shape (N,1) before appling `op`.  This
        has the effect of making broadcasting happen over axis 1 rather than the
        default of axis 0.
    b : ndarray
        If a is 1D shape (P,), convert to shape (N,1) before appling `op`
    op : callable
        binary operation to apply to `a`, `b`

    Returns
    -------
    res : object
        shape equal to ``b.shape``
    """
    shape = b.shape
    if a.ndim == 1:
        a = a[:,None]
    if b.ndim == 1:
        b = b[:,None]
    return op(a, b).reshape(shape)


class AffineError(Exception):
    pass


class affine_transform(object):
    
    def __init__(self, linear_operator, affine_offset, diag=False):
        """ Create affine transform

        Parameters
        ----------
        linear_operator : None or ndarray or sparse array or affine_transform
            Linear part of affine transform implemented as array or as
            affine_transform.  None results in no linear component.
        affine_offset : None or ndarray
            offset component of affine.  Only one of `linear_operator` and
            `affine_offset` can be None, because we need an input array to
            define the shape of the transform.
        diag : {False, True}, optional
            If True, interpret 1D `linear_operator` as the main diagonal of the
            a diagonal array, so that ``linear_operator =
            np.diag(linear_operator)``
        """
        # noneD - linear_operator is None
        # sparseD - linear_operator is sparse
        # affineD - linear_operator is an affine_transform
        # diagD - linear_operator is 1D representation of diagonal
        if linear_operator is None and affine_offset is None:
            raise AffineError('linear_operator and affine_offset cannot '
                              'both be None')
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
        r"""Apply linear part of transform to `x`

        Return :math:`Dx`

        Parameters
        ----------
        x : ndarray
            array to which to apply transform.  Can be 1D or 2D
        copy : {True, False}, optional
            If True, in situations where return is identical to `x`, ensure
            returned value is a copy.

        Returns
        -------
        Dx : ndarray
            `x` transformed with linear component

        Notes
        -----
        This routine is subclassed in affine_atom as a matrix multiplications,
        but could also call FFTs if D is a DFT matrix, in a subclass.
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
        elif self.sparseD:
            return self.linear_operator * x
        elif self.diagD:
            # Deal with 1D or 2D input or linear operator
            return broadcast_first(self.linear_operator, x, mul)
        return np.dot(self.linear_operator, x)

    def affine_map(self, x, copy=True):
        r"""Apply linear and affine offset to `x`

        Return :math:`Dx+\alpha`

        Parameters
        ----------
        x : ndarray
            array to which to apply transform.  Can be 1D or 2D
        copy : {True, False}, optional
            If True, in situations where return is identical to `x`, ensure
            returned value is a copy.

        Returns
        -------
        Dx_a : ndarray
            `x` transformed with linear and offset components

        Notes
        -----
        This routine is subclassed in affine_atom as a matrix multiplications,
        but could also call FFTs if D is a DFT matrix, in a subclass.
        """
        if self.affineD:
            v = self.linear_operator.affine_map(x)
        else:
            v = self.linear_map(x, copy)
        if self.affine_offset is not None:
            # Deal with 1D and 2D input, affine_offset cases
            return broadcast_first(self.affine_offset, v, add)
        # if copy is True, v will already be a copy, so no need to check again
        return v

    def adjoint_map(self, u, copy=True):
        r"""Apply transpose of linear component to `u`

        Return :math:`D^Tu`

        Parameters
        ----------
        u : ndarray
            array to which to apply transposed linear part of transform. Can be
            1D or 2D array
        copy : {True, False}, optional
            If True, in situations where return is identical to `u`, ensure
            returned value is a copy.

        Returns
        -------
        DTu : ndarray
            `u` transformed with transpose of linear component

        Notes
        -----
        This routine is currently a matrix multiplication, but could
        also call FFTs if D is a DFT matrix, in a subclass.
        """
        if self.noneD:
            # this might have to be a copy but we only multiply by D.T when
            # computing gradient -- this currently doesn't happen in seminorm or
            # smoothed_seminorm
            if copy:
                return u.copy()
            return u
        if self.sparseD_csr:
            return self.linear_operator_T * u
        if self.sparseD:
            return self.linear_operator.T * u
        if self.diagD:
            # Deal with 1D or 2D input or linear operator
            return broadcast_first(self.linear_operator, u, mul)
        if self.affineD:
            return self.linear_operator.adjoint_map(u)
        return np.dot(self.linear_operator.T, u)


class linear_transform(affine_transform):
    """ A linear transform is an affine transform with no affine offset
    """
    def __init__(self, linear_operator, diag=False):
        if linear_operator is None:
            raise AffineError('linear_operator cannot be None')
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

