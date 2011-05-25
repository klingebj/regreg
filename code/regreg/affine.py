import numpy as np
from scipy import sparse

class affine_transform(object):
    
    def __init__(self, linear_operator, affine_offset, diag=False):
        self.affine_offset = affine_offset
        self.linear_operator = linear_operator

        if linear_operator is None:
            self.noneD = True
            self.sparseD = False
            self.diagD = False
            self.primal_shape = affine_offset.shape
            self.dual_shape = affine_offset.shape
        else:
            self.noneD = False
            self.sparseD = sparse.isspmatrix(self.linear_operator)
            if linear_operator.ndim == 1 and not diag:
                self.linear_operator = self.linear_operator.reshape((1,-1))
                self.diagD = False
                self.primal_shape = (self.linear_operator.shape[1],)
                self.dual_shape = (1,)
            elif linear_operator.ndim == 1 and diag:
                self.diagD = True
                self.primal_shape = (linear_operator.shape[0],)
                self.dual_shape = (linear_operator.shape[0],)
            else:
                self.primal_shape = (linear_operator.shape[1],)
                self.dual_shape = (linear_operator.shape[0],)
                self.diagD = False

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
                return u * self.linear_operator
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
                                                              
    def affine_objective(self, u):
        if self.affine_offset is not None:
            return np.dot(u, self.affine_offset)
        else:
            return 0

        

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

    def affine_objective(self, u):
        return 0
