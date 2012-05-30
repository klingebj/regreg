from operator import add, mul
import numpy as np
from scipy import sparse
import warnings

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
    
    def __init__(self, linear_operator, affine_offset, diag=False, primal_shape=None):
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

        if sparse.issparse(affine_offset):
            #Convert sparse offset to an array
            self.affine_offset = affine_offset.toarray()
        else:
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
            if self.sparseD and not self.sparseD_csr:
                warnings.warn("Linear operator matrix is sparse, but not csr_matrix. Convert to csr_matrix for faster multiplications!")
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
            elif not primal_shape is None and (len(primal_shape) == 2):
                #Primal shape is a matrix
                self.primal_shape = primal_shape
                self.dual_shape = (linear_operator.shape[0],primal_shape[1])
                self.diagD = False
                self.affineD = False
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

    def offset_map(self, x):
        r"""Apply affine offset to `x`

        Return :math:`x+\alpha`

        Parameters
        ----------
        x : ndarray
            array to which to apply transform.  Can be 1D or 2D

        Returns
        -------
        x_a : ndarray
            `x` transformed with offset components

        """
        if self.affineD:
            v = self.linear_operator.offset_map(x)
        else:
            v = x
        if self.affine_offset is not None:
            # Deal with 1D and 2D input, affine_offset cases
            return broadcast_first(self.affine_offset, v, add)
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
    def __init__(self, linear_operator, diag=False, primal_shape=None):
        if linear_operator is None:
            raise AffineError('linear_operator cannot be None')
        affine_transform.__init__(self, linear_operator, None, diag=diag, primal_shape=primal_shape)


class selector(linear_transform):

    """
    Apply an affine transform after applying an
    indexing operation to the array.

    >>> X = np.arange(30).reshape((6,5))
    >>> offset = np.arange(6)
    >>> transform = affine_transform(X, offset)
    >>> apply_to_first5 = selector(slice(0,5), (20,), transform)
    >>> apply_to_first5.linear_map(np.arange(20))
    array([ 30,  80, 130, 180, 230, 280])
    >>> np.dot(X, np.arange(5))
    array([ 30,  80, 130, 180, 230, 280])

    >>> apply_to_first5.affine_map(np.arange(20))
    array([ 30,  81, 132, 183, 234, 285])
    >>> np.dot(X, np.arange(5)) + offset
    array([ 30,  81, 132, 183, 234, 285])

    >>> apply_to_first5.adjoint_map(np.arange(6))

    array([ 275.,  290.,  305.,  320.,  335.,    0.,    0.,    0.,    0.,
              0.,    0.,    0.,    0.,    0.,    0.,    0.,    0.,    0.,
              0.,    0.])

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

    def linear_map(self, x, copy=False):
        x_indexed = x[self.index_obj]
        return self.affine_transform.linear_map(x_indexed)

    def affine_map(self, x, copy=False):
        x_indexed = x[self.index_obj]
        return self.affine_transform.affine_map(x_indexed)

    def offset_map(self, x, copy=False):
        x_indexed = x[self.index_obj]
        return self.affine_transform.offset_map(x_indexed)

    def adjoint_map(self, u, copy=False):
        if not hasattr(self, "_output"):
            self._output = np.zeros(self.initial_shape)
        self._output[self.index_obj] = self.affine_transform.adjoint_map(u)
        return self._output

class reshape(linear_transform):

    """
    Reshape the output of an affine transform.

    """

    def __init__(self, primal_shape, dual_shape):
        self.primal_shape = primal_shape
        self.dual_shape = dual_shape

    def linear_map(self, x, copy=False):
        if copy:
            x = x.copy()
        return x.reshape(self.dual_shape)

    def affine_map(self, x, copy=False):
        return self.linear_map(x, copy)

    def offset_map(self, x, copy=False):
        if copy:
            x = x.copy()
        return x

    def adjoint_map(self, u, copy=True):
        if copy:
            u = u.copy()
        return u.reshape(self.dual_shape)

def tensor(T, first_primal_index):
    primal_shape = T.shape[first_primal_index:]
    dual_shape = T.shape[:first_primal_index]

    Tm = T.reshape((np.product(dual_shape),
                    np.product(primal_shape)))
    reshape_primal = reshape(primal_shape, Tm.shape[1])
    reshape_dual = reshape(Tm.shape[0], dual_shape)
    return composition(reshape_dual, Tm, reshape_primal)

class normalize(object):

    '''
    Normalize column by means and possibly scale. Could make
    a class for row normalization to.

    Columns are normalized to have std equal to value.
    '''

    def __init__(self, M, center=True, scale=True, value=1, inplace=False,
                 intercept_column=None):
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
            If sensible, modify values in place. For a sparse matrix,
            this will raise an exception if True and center==True.

        intercept_column : [int,None]
            Which column is the intercept if any? This column is
            not centered or scaled.

        '''
        n, p = M.shape
        self.value = value
        self.dual_shape = (n,)
        self.primal_shape = (p,)

        self.sparseM = sparse.isspmatrix(M)
        self.intercept_column = intercept_column
        self.M = M

        self.center = center
        self.scale = scale

        if value != 1 and not scale:
            raise ValueError('setting value when not being asked to scale')

        # we divide by n instead of n-1 in the scalings
        # so that np.std is constant
        
        if self.center:
            if inplace and self.sparseM:
                raise ValueError('resulting matrix will not be sparse if centering performed inplace')

            if not self.sparseM:
                col_means = M.mean(0)
            else:
                tmp = M.copy()
                col_means = np.asarray(tmp.mean(0)).reshape(-1)
                tmp.data **= 2

            if self.intercept_column is not None:
                col_means[self.intercept_column] = 0

            if self.scale:
                if not self.sparseM:
                    self.col_stds = np.sqrt((np.sum(M**2,0) - n * col_means**2) / n) / np.sqrt(self.value)
                else:
                    self.col_stds = np.sqrt((np.asarray(tmp.sum(0)).reshape(-1) - n * col_means**2) / n) / np.sqrt(self.value)
                if self.intercept_column is not None:
                    self.col_stds[self.intercept_column] = 1. / np.sqrt(self.value)

            if not self.sparseM and inplace:
                self.M -= col_means[np.newaxis,:]
                if self.scale:
                    self.M /= self.col_stds[np.newaxis,:]
                    # if scaling has been applied in place, 
                    # no need to do it again
                    self.col_stds = None
                    self.scale = False
        elif self.scale:
            if not self.sparseM:
                self.col_stds = np.sqrt(np.sum(M**2,0) / n) / np.sqrt(self.value)
            else:
                tmp = M.copy()
                tmp.data **= 2
                self.col_stds = np.sqrt((tmp.sum(0)) / n) / np.sqrt(self.value)
            if self.intercept_column is not None:
                self.col_stds[self.intercept_column] = 1. / np.sqrt(self.value)
            if inplace:
                self.M /= self.col_stds[np.newaxis,:]
                # if scaling has been applied in place, 
                # no need to do it again
                self.col_stds = None
                self.scale = False
        self.affine_offset = None
        
    def linear_map(self, x):
        shift = None
        if self.intercept_column is not None:
            if self.scale and self.value != 1:
                x_intercept = x[self.intercept_column] * np.sqrt(self.value)
            else:
                x_intercept = x[self.intercept_column]
        if self.scale:
            if x.ndim == 1:
                x = x / self.col_stds
            elif x.ndim == 2:
                x = x / self.col_stds[:,np.newaxis]
            else:
                raise ValueError('normalize only implemented for 1D and 2D inputs')
        if self.sparseM:
            v = self.M * x
        else:
            v = np.dot(self.M, x)
        if self.center:
            if x.ndim == 1:
                v -= v.mean()
            elif x.ndim == 2:
                v -= v.mean(0)[np.newaxis,:]
            else:
                raise ValueError('normalize only implemented for 1D and 2D inputs')
        if shift is not None: # what is this shift possibly for?
            if x.ndim == 1:
                return v + shift
            elif x.ndim == 2:
                w = v + shift[np.newaxis,:]
            else:
                raise ValueError('normalize only implemented for 1D and 2D inputs')
        if self.intercept_column is not None and self.center:
            if x.ndim == 2:
                v += x_intercept[np.newaxis,:]
            else:
                v += x_intercept
        return v

    def affine_map(self, x):
        return self.linear_map(x)

    def offset_map(self, x):
        return x

    def adjoint_map(self, u):
        v = np.empty(self.primal_shape)
        if self.center:
            if u.ndim == 1:
                u_mean = u.mean()
                u = u - u_mean
            elif u.ndim == 2:
                u_mean = u.mean(0)
                u = u - u_mean[np.newaxis,:]
            else:
                raise ValueError('normalize only implemented for 1D and 2D inputs')
        if self.sparseM:
            v = (u.T * self.M).T
        else:
            v = np.dot(u.T, self.M).T
        if self.scale:
            v /= self.col_stds
        if self.intercept_column is not None:
            v[self.intercept_column] = u_mean * u.shape[0]
        return v

    def slice_columns(self, index_obj):
        """

        Parameters
        ----------

        index_obj: slice, list, np.bool
            An object on which to index the columns of self.M.
            Must be a slice object or list so scipy.sparse matrices
            can be sliced.

        Returns
        -------
        
        n : normalize
            A transform which agrees with self having zeroed out
            all coefficients not captured by index_obj.

        Notes
        -----

        This method does not check whether or not self.intercept_column
        is None.
        
        >>> X = np.array([1.2,3.4,5.6,7.8,1.3,4.5,5.6,7.8,1.1,3.4])
        >>> D = np.identity(X.shape[0]) - np.diag(np.ones(X.shape[0]-1),1)
        >>> nD = ra.normalize(D)
        >>> X_sliced = X.copy()
        >>> X_sliced[:4] = 0; X_sliced[6:] = 0

        >>> nD.linear_map(X_sliced)
        array([  0.        ,   0.        ,   0.        ,  -2.90688837,
                -7.15541753,  10.0623059 ,   0.        ,   0.        ,
                 0.        ,   0.        ])

        >>> nD_slice = nD.slice_columns(slice(4,6))
        >>> nD_slice.linear_map(X[slice(4,6)])
        array([  0.        ,   0.        ,   0.        ,  -2.90688837,
                -7.15541753,  10.0623059 ,   0.        ,   0.        ,
                 0.        ,   0.        ])
        >>> 

        
        """
        if type(index_obj) not in [type(slice(0,4)), type([])]:
            # try to find nonzero indices if a boolean array
            if index_obj.dtype == np.bool:
                index_obj = np.nonzero(index_obj)[0]
        
        new_obj = normalize.__new__(normalize)
        new_obj.sparseM = self.sparseM

        new_obj.M = self.M[:,index_obj]

        new_obj.primal_shape = (new_obj.M.shape[1],)
        new_obj.dual_shape = (self.M.shape[0],)
        new_obj.scale = self.scale
        new_obj.center = self.center
        if self.scale:
            new_obj.col_stds = self.col_stds[index_obj]
        new_obj.affine_offset = self.affine_offset
        return new_obj
        
class identity(object):

    def __init__(self, primal_shape):
        self.primal_shape = self.dual_shape = primal_shape
        self.affine_offset = None
        self.linear_operator = None

    def affine_map(self, x, copy=True):
        return self.linear_map(x, copy)

    def offset_map(self, x, copy=True):
        if copy:
            return x.copy()
        else:
            return x

    def linear_map(self, x, copy=True):
        if copy:
            return x.copy()
        else:
            return x

    def adjoint_map(self, x, copy=True):
        return self.linear_map(x, copy)

class vstack(object):
    """
    Stack several affine transforms vertically together though
    not necessarily as a big matrix.
   
    """

    def __init__(self, transforms):
        self.primal_shape = -1
        self.dual_shapes = []
        self.transforms = []
        self.dual_slices = []
        total_dual = 0
        for transform in transforms:
            transform = astransform(transform)
            if self.primal_shape == -1:
                self.primal_shape = transform.primal_shape
            else:
                if transform.primal_shape != self.primal_shape:
                    raise ValueError("primal dimensions don't agree")
            self.transforms.append(transform)
            self.dual_shapes.append(transform.dual_shape)
            increment = np.product(transform.dual_shape)
            self.dual_slices.append(slice(total_dual, total_dual + increment))
            total_dual += increment

        self.dual_shape = (total_dual,)
        self.group_dtype = np.dtype([('group_%d' % i, np.float, shape) 
                                     for i, shape in enumerate(self.dual_shapes)])
        self.dual_groups = self.group_dtype.names 

        # figure out the affine offset
        self.affine_offset = np.empty(self.dual_shape)
        x = np.zeros(self.primal_shape)
        for g, t in zip(self.dual_slices, self.transforms):
            self.affine_offset[g] = t.affine_map(x)
        if np.all(np.equal(self.affine_offset, 0)):
            self.affine_offset = None
            
    def linear_map(self, x, copy=False):
        result = np.empty(self.dual_shape)
        for g, t in zip(self.dual_slices, self.transforms):
            result[g] = t.linear_map(x)
        return result

    def affine_map(self, x, copy=False):
        result = np.empty(self.dual_shape)
        for g, t in zip(self.dual_slices, self.transforms):
            result[g] = t.linear_map(x)
        if self.affine_offset is not None:
            return result + self.affine_offset
        else:
            return result

    def offset_map(self, x, copy=False):
        if self.affine_offset is not None:
            return x + self.affine_offset
        else:
            return x

    def adjoint_map(self, u, copy=False):
        result = np.zeros(self.primal_shape)
        for g, t, s in zip(self.dual_slices, self.transforms,
                           self.dual_shapes):
            result += t.adjoint_map(u[g].reshape(s))
        return result

class hstack(object):
    """
    Stack several affine transforms horizontally together though
    not necessarily as a big matrix.
   
    """

    def __init__(self, transforms):
        self.dual_shape = -1
        self.primal_shapes = []
        self.transforms = []
        self.primal_slices = []
        total_primal = 0
        for transform in transforms:
            transform = astransform(transform)
            if self.dual_shape == -1:
                self.dual_shape = transform.dual_shape
            else:
                if transform.dual_shape != self.dual_shape:
                    raise ValueError("dual dimensions don't agree")
            self.transforms.append(transform)
            self.primal_shapes.append(transform.primal_shape)
            increment = np.product(transform.primal_shape)
            self.primal_slices.append(slice(total_primal, total_primal + increment))
            total_primal += increment

        self.primal_shape = (total_primal,)
        self.group_dtype = np.dtype([('group_%d' % i, np.float, shape) 
                                     for i, shape in enumerate(self.primal_shapes)])
        self.primal_groups = self.group_dtype.names 

        # figure out the affine offset
        self.affine_offset = np.zeros(self.dual_shape)
        for g, s, t in zip(self.primal_slices, self.primal_shapes,
                           self.transforms):
            self.affine_offset += t.affine_map(np.zeros(s))
        if np.all(np.equal(self.affine_offset, 0)):
            self.affine_offset = None

    def linear_map(self, x, copy=False):
        result = np.zeros(self.dual_shape)
        for g, t, s in zip(self.primal_slices, self.transforms,
                           self.primal_shapes):
            result += t.linear_map(x[g].reshape(s))
        return result

    def affine_map(self, x, copy=False):
        result = np.zeros(self.dual_shape)
        for g, t, s in zip(self.primal_slices, self.transforms,
                        self.primal_shapes):
            result += t.linear_map(x[g].reshape(s))
        if self.affine_offset is not None:
            return result + self.affine_offset
        else:
            return result

    def offset_map(self, x, copy=False):
        if self.affine_offset is not None:
            return x + self.affine_offset
        else:
            return x

    def adjoint_map(self, u, copy=False):
        result = np.empty(self.primal_shape)
        #XXX this reshaping will fail for shapes that aren't
        # 1D, would have to view as self.group_dtype to
        # take advantange of different shapes
        for g, t, s in zip(self.primal_slices, self.transforms,
                           self.primal_shapes):
            result[g] = t.adjoint_map(u).reshape(-1)
        return result

def power_L(transform, max_its=500,tol=1e-8, debug=False):
    """
    Approximate the largest singular value (squared) of the linear part of
    a transform using power iterations
    
    TODO: should this be the largest singular value instead (i.e. not squared?)
    """

    transform = astransform(transform)
    v = np.random.standard_normal(transform.primal_shape)
    old_norm = 0.
    norm = 1.
    itercount = 0
    while np.fabs(norm-old_norm)/norm > tol and itercount < max_its:
        v = transform.adjoint_map(transform.linear_map(v))
        old_norm = norm
        norm = np.linalg.norm(v)
        v /= norm
        if debug:
            print "L", norm
        itercount += 1
    return norm

def astransform(X):
    """
    If X is an affine_transform, return X,
    else try to cast it as an affine_transform
    """
    if isinstance(X, affine_transform):
        return X
    else:
        return linear_transform(X)

class adjoint(object):

    """
    Given an affine_transform, return a linear_transform
    that is the adjoint of its linear part.
    """
    def __init__(self, transform):
        self.transform = astransform(transform)
        self.affine_offset = None
        self.primal_shape = self.transform.dual_shape
        self.dual_shape = self.transform.primal_shape

    def linear_map(self, x, copy=False):
        return self.transform.adjoint_map(x, copy)

    def affine_map(self, x, copy=False):
        return self.linear_map(x, copy)

    def offset_map(self, x, copy=False):
        return x

    def adjoint_map(self, x, copy=False):
        return self.transform.linear_map(x, copy)

class tensorize(object):

    """
    Given an affine_transform, return a linear_transform
    that expects q copies of something with transform's primal_shape.

    This class effectively makes explicit that a transform
    may expect a matrix rather than a single vector.

    """
    def __init__(self, transform, q):
        self.transform = astransform(transform)
        self.affine_offset = self.transform.affine_offset
        self.primal_shape = self.transform.primal_shape + (q,)
        self.dual_shape = self.transform.dual_shape + (q,)

    def linear_map(self, x):
        return self.transform.linear_map(x)

    def affine_map(self, x):
        v = self.linear_map(x) 
        if self.affine_offset is not None:
            return v + self.affine_offset[:, np.newaxis]
        return v

    def offset_map(self, x):
        return x

    def adjoint_map(self, x):
        return self.transform.adjoint_map(x)

class residual(object):

    """
    Compute the residual from an affine transform.
    """

    def __init__(self, transform):
        self.transform = astransform(transform)
        self.primal_shape = self.transform.primal_shape
        self.dual_shape = self.transform.dual_shape
        self.affine_offset = None
        if not self.primal_shape == self.dual_shape:
            raise ValueError('dual and primal shapes should be the same to compute residual')

    def linear_map(self, x):
        return x - self.transform.linear_map(x)

    def affine_map(self, x):
        return x - self.transform.affine_map(x)

    def adjoint_map(self, u):
        return u - self.transform.adjoint_map(u)

class composition(object):

    """
    Composes a list of affine transforms, executing right to left
    """

    def __init__(self, *transforms):
        self.transforms = [astransform(t) for t in transforms]
        self.primal_shape = self.transforms[-1].primal_shape
        self.dual_shape = self.transforms[0].dual_shape

        # compute the affine_offset
        affine_offset = self.affine_map(np.zeros(self.primal_shape))
        if not np.allclose(affine_offset, 0): 
            self.affine_offset = None
        else:
            self.affine_offset = affine_offset

    def linear_map(self, x):
        output = x
        for transform in self.transforms[::-1]:
            output = transform.linear_map(output)
        return output

    def affine_map(self, x):
        output = x
        for transform in self.transforms[::-1]:
            output = transform.affine_map(output)
        return output

    def offset_map(self, x):
        output = x
        for transform in self.transforms[::-1]:
            output = transform.offset_map(output)
        return output

    def adjoint_map(self, x):
        output = x
        for transform in self.transforms:
            output = transform.adjoint_map(output)
        return output

class affine_sum(object):

    """
    Creates the (weighted) sum of a list of affine_transforms
    """

    def __init__(self, transforms, weights=None):
        self.transforms = transforms
        if weights is None:
            self.weights = np.ones(len(self.transforms))
        else:
            if not len(self.transforms) == len(weights):
                raise ValueError("Must specify a weight for each transform")
            self.weights = weights
        self.primal_shape = transforms[0].primal_shape
        self.dual_shape = transforms[0].dual_shape

        # compute the affine_offset
        affine_offset = self.affine_map(np.zeros(self.primal_shape))
        if np.allclose(affine_offset, 0): 
            self.affine_offset = None
        else:
            self.affine_offset = affine_offset

    def linear_map(self, x):
        output = 0
        for transform, weight in zip(self.transforms[::-1], self.weights[::-1]):
            output += weight * transform.linear_map(x)
        return output

    def affine_map(self, x):
        output = 0
        for transform, weight in zip(self.transforms[::-1], self.weights[::-1]):
            output += weight * transform.affine_map(x)
        return output


    def offset_map(self, x):
        return self.affine_offset

    def adjoint_map(self, x):
        output = 0
        for transform, weight in zip(self.transforms[::-1], self.weights[::-1]):
            output += weight * transform.adjoint_map(x)
        return output

def difference_transform(X, order=1, sorted=False,
                         transform=False):
    """
    Compute the divided difference matrix for X
    after sorting X.

    Parameters
    ----------

    X: np.array, np.float, ndim=1

    order: int
        What order of difference should we compute?

    sorted: bool
        Is X sorted?

    transform: bool
        If True, return a linear_transform rather
        than a sparse matrix.

    Returns
    -------

    D: np.array, ndim=2, shape=(n-order,order)
        Matrix of divided differences of sorted X.

    """
    if not sorted:
        X = np.sort(X)
    X = np.asarray(X)
    n = X.shape[0]
    Dfinal = np.identity(n)
    for j in range(1, order+1):
        D = (-np.identity(n-j+1)+np.diag(np.ones(n-j),k=1))[:-1]
        steps = X[j:]-X[:-j]
        inv_steps = np.zeros(steps.shape)
        inv_steps[steps != 0] = 1. / steps[steps != 0]
        D = np.dot(np.diag(inv_steps), D)
        Dfinal = np.dot(D, Dfinal)
    if not transform:
        return sparse.csr_matrix(Dfinal)
    return astransform(Dfinal)


class scalar_multiply(object):

    def __init__(self, atransform, scalar):
        self.primal_shape, self.dual_shape = (atransform.primal_shape, atransform.dual_shape)
        self.scalar = scalar
        self.affine_offset = None
        self._atransform = atransform

    def affine_map(self, x, copy=True):
        if self.scalar != 1.:
            return self._atransform.linear_map(x, copy) * self.scalar
        else:
            return self._atransform.linear_map(x, copy)

    def offset_map(self, x, copy=True):
        if self.scalar != 1.:
            return self._atransform.offset_map(x, copy) * self.scalar # is this correct -- what is offset_map again?
        else:
            return self._atransform.offset_map(x, copy)

    def linear_map(self, x, copy=True):
        if self.scalar != 1.:
            return self._atransform.linear_map(x, copy) * self.scalar 
        else:
            return self._atransform.linear_map(x, copy)


    def adjoint_map(self, x, copy=True):
        if self.scalar != 1.:
            return self._atransform.adjoint_map(x, copy) * self.scalar 
        else:
            return self._atransform.adjoint_map(x, copy)
