import numpy as np
import regreg.api as rr
from regreg.affine import composition, hstack

class symm_from_lower(object):

    def __init__(self, dim):
        self.primal_shape = dim*(dim+1)/2
        self.dual_shape = (dim,dim)
        self.affine_offset = None
        self.dim = dim

    def linear_map(self, x):
        result = np.zeros(self.dual_shape)
        c = 0
        for i in range(self.dim):
            for j in range(i+1):
                result[i,j] = result[j,i] = x[c]
                c += 1
        return result

    def affine_map(self, x):
        return self.linear_map(x)

    def adjoint_map(self, x):
        result = np.zeros(self.primal_shape)
        c = 0
        for i in range(self.dim):
            for j in range(i+1):
                result[c] = x[i,j]
                c += 1
        return result

class outer(object):

    def __init__(self, dim):
        self.dim = dim
        self.primal_shape = dim
        self.dual_shape = dim*(dim+1)/2
        self.affine_offset = None
        self.L = symm_lower(dim)

    def linear_map(self, x):
        return self.L.adjoint_map(np.add.outer(x,x))

    def affine_map(self, x):
        return self.linear_map(x)

    def adjoint_map(self, x):
        V = self.L.linear_map(x)
        return np.sum(V,0) + np.sum(V,1)

class interaction_diff(object):

    def __init__(self, dim):
        self.dim = dim
        self.primal_shape = dim + dim*(dim+1)/2
        self.dual_shape = dim*(dim+1)/2
        self.L = outer(dim)
        self.affine_offset = None

    def linear_map(self, x):
        # the first dim coordinates are \beta,
        # the remaining are \Theta
        v = self.L.linear_map(x[:self.dim])
        return x[self.dim:] - v

    def affine_map(self, x):
        return self.linear_map(x)

    def adjoint_map(self, x):
        result = np.zeros(self.primal_shape)
        result[self.dim:] = x
        result[:self.dim] = -self.L.adjoint_map(x)
        return result

class interaction_sum(interaction_diff):

    def linear_map(self, x):
        # the first dim coordinates are \beta,
        # the remaining are \Theta
        v = self.L.linear_map(x[:self.dim])
        return x[self.dim:] + v

    def affine_map(self, x):
        return self.linear_map(x)

    def adjoint_map(self, x):
        result = np.zeros(self.primal_shape)
        result[self.dim:] = x
        result[:self.dim] = self.L.adjoint_map(x)
        return result

class posneg(interaction_diff):

    def __init__(self, dim):
        self.dim = dim
        self.primal_shape = 3*dim
        self.dual_shape = dim
        self.affine_offset = None

    def linear_map(self, x):
        return x[:self.dim] - x[self.dim:(2*self.dim)] - x[(2*self.dim):]

    def affine_map(self, x):
        return self.linear_map(x)

    def adjoint_map(self, x):
        result = np.zeros(self.primal_shape)
        result[:self.dim] = x
        result[self.dim:(2*self.dim)] = -x
        result[(2*self.dim):] = -x
        return result

if __name__ == "__main__":

    def posneg_interaction(i):
        return hstack([interaction_diff(i),interaction_diff(i)])

    import pylab
    interaction_L=[power_L(posneg_interaction(i)) for i in range(1,20)]
    pylab.plot(interaction_L)
    pylab.plot(8*np.arange(1,20)+2)
    pylab.show()

    posneg_part = np.array([power_L(posneg(i)) for i in range(1,20)])
    print np.fabs(posneg_part-3).max()


