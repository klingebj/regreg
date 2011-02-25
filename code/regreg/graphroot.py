import numpy as np

# Local imports

import subfunctions as sf
import updates
import l1smooth
from problems import linmodel

class graphroot(linmodel):

    """
    GraphRoot problem:
    Minimizes the problem

    .. math::

       \begin{eqnarray}
       \|y - X\beta\|^{2}_{2}/2 + \lambda_{1}\|\beta\|_{1} + \lambda_2 h_\mu(D\beta)
       \end{eqnarray}

    where

    .. math::

       \begin{eqnarray}
       h_\mu(x) = \left\{\begin{array} \|Dx\| - \mu/2 &\mbox{ if } \|Dx\| \geq \mu \\ \frac{1}{2\mu}\|Dx\|^2 & \mbox{ else} \end{array} \right.
       \end{eqnarray}
    as a function of beta. This is a smooth approximation to minimizing

    .. math::

       \begin{eqnarray}
       \|y - X\beta\|^{2}_{2}/2 + \lambda_{1}\|\beta\|_{1} + \lambda_2 \sqrt( \beta^T L \beta )
       \end{eqnarray}
    
    """

    def initialize(self, data):
        """
        Generate initial tuple of arguments for update.
        """

        if len(data) == 3:
            self.X = data[0]
            self.Y = data[1]
            self.D = data[2]
            self.n, self.p = self.X.shape
        else:
            raise ValueError("Data tuple not as expected")


        if hasattr(self,'initial_coefs'):
            self.set_coefs(self.initial_coefs)
        else:
            self.set_coefs(self.default_coefs)

            
    @property
    def default_penalties(self):
        """
        Default penalties for GraphRoot
        """
        default = np.zeros(1, np.dtype([(l, np.float) for l in ['l1', 'l2','mu']]))
        default['mu'] = 1.
        return default
    

    def obj(self, beta):
        beta = np.asarray(beta)
        return self.obj_smooth(beta) + self.obj_rough(beta)

    def obj_smooth(self, beta):
        #Smooth part of objective
        u = np.dot(self.D,beta)
        if self.penalties['mu'] <= np.linalg.norm(u):
            return ((self.Y - np.dot(self.X, beta))**2).sum() / 2. +  np.linalg.norm(u) * self.penalties['l2'] - self.penalties['mu']/2.
        else:
            return ((self.Y - np.dot(self.X, beta))**2).sum() / 2. + np.linalg.norm(u)**2 * self.penalties['l2']/(2. * self.penalties['mu'])

    def obj_rough(self, beta):
        return np.sum(np.fabs(beta)) * self.penalties['l1']  


class gengrad(graphroot):
    
    def grad(self, beta):
        beta = np.asarray(beta)
        XtXbeta = np.dot(self.X.T, np.dot(self.X, beta))
        u = np.dot(self.D,beta)
        if self.penalties['mu'] <= np.linalg.norm(u):
            return XtXbeta - np.dot(self.Y,self.X) + self.penalties['l2']*np.dot(self.D.T,u)/np.linalg.norm(u)  
        else:
            return XtXbeta - np.dot(self.Y,self.X) + self.penalties['l2']*np.dot(self.D.T,u)/self.penalties['mu']

    def proximal(self, z, g, L):
        v = z - g / L
        return np.sign(v) * np.maximum(np.fabs(v)-self.penalties['l1']/L, 0)


class graphroot_sparse(graphroot):
    
    #Assume Lap given as scipy.sparse matrix
    def obj_smooth(self, beta):
        beta = np.asarray(beta)
        u = self.D.matvec(beta)
        if self.penalties['mu'] <= np.linalg.norm(u):
            return ((self.Y - np.dot(self.X, beta))**2).sum() / 2. +  np.linalg.norm(u) * self.penalties['l2'] - self.penalties['mu']/2.
        else:
            return ((self.Y - np.dot(self.X, beta))**2).sum() / 2. + np.linalg.norm(u)**2 * self.penalties['l2']/(2. * self.penalties['mu'])

class gengrad_sparse(graphroot_sparse):
    
    def grad(self, beta):
        beta = np.asarray(beta)
        XtXbeta = np.dot(self.X.T, np.dot(self.X, beta)) 
        u = self.D.matvec(beta)
        if self.penalties['mu'] <= np.linalg.norm(u):
            return XtXbeta - np.dot(self.Y,self.X) + self.penalties['l2']*self.D.T.matvec(u)/np.linalg.norm(u)  
        else:
            return XtXbeta - np.dot(self.Y,self.X) + self.penalties['l2']*self.D.T.matvec(u)/self.penalties['mu']

    def proximal(self, z, g, L):
        v = z - g / L
        return np.sign(v) * np.maximum(np.fabs(v)-self.penalties['l1']/L, 0)

         
