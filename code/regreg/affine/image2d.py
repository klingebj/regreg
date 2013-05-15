import numpy as np
from scipy import sparse
from ..affine import affine_transform

def formD_smaller(m, n):
    """
    Generate a sparse
    matrix that computes
    the difference across all edges in a
    2D lattice of shape (m,n).

    Parameters
    ----------

    m: int

    n: int

    Returns
    -------

    D: scipy.csr_matrix
       A matrix with 2*(m-1)*(n-1)+(m-1)+(n-1) rows
       and m*n columns
       representing each edge exactly once in the
       2D lattice.

       The first 2*(m-1)*(n-1) rows represent
       interior vertices, then the right hand edges and
       finally the top edges with the origin in the bottom left.
    
    """
    p = m*n

    # interior vertices that have
    # both a vertex to their right and above them
    # (starting from bottom left)

    vertices = np.indices((m-1,n-1)).reshape((2, -1)).T
    strides = np.empty((m,n), np.bool).strides
    # D = sparse.lil_matrix((2*(m-1)*(n-1)+(m-1)+(n-1),p))
    
    idx = 0
    vertices_in_strides = np.dot(vertices, strides)
    vertices_in_strides_shift1 = vertices_in_strides + np.dot([1,0], strides)
    vertices_in_strides_shift2 = vertices_in_strides + np.dot([0,1], strides)

    indices = np.array([np.arange(vertices.shape[0]) + idx,
                        vertices_in_strides])

    D1 = sparse.csr_matrix((np.ones(vertices.shape[0]),
                            indices), (2*(m-1)*(n-1)+(m-1)+(n-1),p))
    indices[1] = vertices_in_strides_shift1
    D2 = sparse.csr_matrix((-np.ones(vertices.shape[0]),
                            indices), (2*(m-1)*(n-1)+(m-1)+(n-1),p))
    idx += vertices.shape[0]

    indices = np.array([np.arange(vertices.shape[0]) + idx,
                        vertices_in_strides])
    D3 = sparse.csr_matrix((np.ones(vertices.shape[0]),
                            indices), (2*(m-1)*(n-1)+(m-1)+(n-1),p))
    indices[1] = vertices_in_strides_shift2
    D4 = sparse.csr_matrix((-np.ones(vertices.shape[0]),
                            indices), (2*(m-1)*(n-1)+(m-1)+(n-1),p))

    idx += vertices.shape[0]

    # Now, do the top edges for each
    # vertex in the right edge

    vertices = np.indices((1,n-1)).reshape((2, -1)).T + np.array([m-1,0])
    vertices_in_strides = np.dot(vertices, strides)
    vertices_in_strides_shift3 = vertices_in_strides + np.dot([0,1], strides)

    indices = np.array([np.arange(vertices.shape[0]) + idx,
                        vertices_in_strides])
    D5 = sparse.csr_matrix((np.ones(vertices.shape[0]),
                            indices), (2*(m-1)*(n-1)+(m-1)+(n-1),p))
    indices[1] = vertices_in_strides_shift3
    D6 = sparse.csr_matrix((-np.ones(vertices.shape[0]),
                            indices), (2*(m-1)*(n-1)+(m-1)+(n-1),p))
    idx += vertices.shape[0]

    # Finally, the top edge
    
    vertices = np.indices((m-1,1)).reshape((2, -1)).T + np.array([0,n-1])
    vertices_in_strides = np.dot(vertices, strides)
    vertices_in_strides_shift4 = vertices_in_strides + np.dot([1,0], strides)

    indices = np.array([np.arange(vertices.shape[0]) + idx,
                        vertices_in_strides])
    D7 = sparse.csr_matrix((np.ones(vertices.shape[0]),
                            indices), (2*(m-1)*(n-1)+(m-1)+(n-1),p))
    indices[1] = vertices_in_strides_shift4
    D8 = sparse.csr_matrix((-np.ones(vertices.shape[0]),
                            indices), (2*(m-1)*(n-1)+(m-1)+(n-1),p))

    return D1 + D2 + D3 + D4 + D5 + D6 + D7 + D8


def formD(m, n):
    """
    Generate a sparse
    matrix that computes
    the difference across all edges in a
    2D lattice of shape (m,n).

    Parameters
    ----------
    m: int

    n: int

    Returns
    -------

    D: scipy.csr_matrix
       A matrix with 2*(m*n-1)
       and m*n columns
       representing each edge exactly once in the
       2D lattice and including some zero rows.

       This shape is used because it associates
       exactly two edges to each vertex except
       the top right. Hence, the differences
       can be formed into a matrix with two rows
       to which the l1_l2 proximal function
       can be applied.

       The first (m-1)*(n-1) rows represent
       interior vertices, then the right hand edges and
       finally the top edges with the origin in the bottom left.
       This forms the first row of this two row matrix described above.

       The second row of the matrix has exactly the same structure.

    """

    p = m*n

    # interior vertices that have
    # both a vertex to their right and above them
    # (starting from bottom left)

    vertices = np.indices((m-1,n-1)).reshape((2, -1)).T
    strides = np.empty((m,n), np.bool).strides
    nrow = 2*(m*n-1)
    
    idx = 0
    vertices_in_strides = np.dot(vertices, strides)
    vertices_in_strides_shift1 = vertices_in_strides + np.dot([1,0], strides)

    indices = np.array([np.arange(vertices.shape[0]) + idx,
                        vertices_in_strides])

    D1 = sparse.csr_matrix((np.ones(vertices.shape[0]),
                            indices), (nrow,p))
    indices[1] = vertices_in_strides_shift1
    D2 = sparse.csr_matrix((-np.ones(vertices.shape[0]),
                            indices), (nrow,p))
    idx += vertices.shape[0]

    # now, do the top edge

    vertices = np.indices((1,n-1)).reshape((2, -1)).T + np.array([m-1,0])
    vertices_in_strides = np.dot(vertices, strides)
    vertices_in_strides_shift3 = vertices_in_strides + np.dot([0,1], strides)

    indices = np.array([np.arange(vertices.shape[0]) + idx,
                        vertices_in_strides])
    D5 = sparse.csr_matrix((np.ones(vertices.shape[0]),
                            indices), (nrow,p))
    indices[1] = vertices_in_strides_shift3
    D6 = sparse.csr_matrix((-np.ones(vertices.shape[0]),
                            indices), (nrow,p))
    idx += vertices.shape[0] 

    # Finally, the top edge
    
    vertices = np.indices((m-1,1)).reshape((2, -1)).T + np.array([0,n-1])
    vertices_in_strides = np.dot(vertices, strides)
    vertices_in_strides_shift4 = vertices_in_strides + np.dot([1,0], strides)

    indices = np.array([np.arange(vertices.shape[0]) + idx,
                        vertices_in_strides])
    D7 = sparse.csr_matrix((np.ones(vertices.shape[0]),
                            indices), (nrow,p))
    indices[1] = vertices_in_strides_shift4
    D8 = sparse.csr_matrix((-np.ones(vertices.shape[0]),
                            indices), (nrow,p))
    idx += vertices.shape[0]

    # this is the second 'row' of the tensor, which is
    # nonzero only for interior vertices

    vertices = np.indices((m-1,n-1)).reshape((2, -1)).T
    vertices_in_strides = np.dot(vertices, strides)
    vertices_in_strides_shift2 = vertices_in_strides + np.dot([0,1], strides)

    indices = np.array([np.arange(vertices.shape[0]) + idx,
                        vertices_in_strides])
    D3 = sparse.csr_matrix((np.ones(vertices.shape[0]),
                            indices), (nrow,p))
    indices[1] = vertices_in_strides_shift2
    D4 = sparse.csr_matrix((-np.ones(vertices.shape[0]),
                            indices), (nrow,p))


    return D1 + D2 + D3 + D4 + D5 + D6 + D7 + D8

class image2d_differences(affine_transform):

    def __init__(self, image_shape, affine_offset=None):
        self.image_shape = image_shape
        self.D = formD(*image_shape)
        self.DT = self.D.T.tocsr()
        self.input_shape = image_shape
        m, n = self.input_shape
        self.output_shape = (m*n-1,2)
        self.affine_offset = affine_offset

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
        """
        v = self.D * x.reshape(-1)
        return v.reshape(self.output_shape[::-1]).T

    def affine_map(self, x, copy=True):
        r"""Apply linear part of transform to `x`

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
        Dx + offset: ndarray
            `x` transformed with linear component + offset
        """
        if self.affine_offset is not None:
            return self.linear_map(x) + self.affine_offset
        return self.linear_map(x)

    def adjoint_map(self, u, copy=True):
        r"""Apply adjioint of transform to `u`

        Return :math:`D^Tu`

        Parameters
        ----------
        u : ndarray
            array to which to apply transform.  Can be 1D or 2D
        copy : {True, False}, optional
            If True, in situations where return is identical to ``, ensure
            returned value is a copy.

        Returns
        -------
        D.T*u : ndarray
            `u` transformed with linear component
        """
        v = self.DT * u.T.reshape(-1)
        return v.reshape(self.input_shape)
