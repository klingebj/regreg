import numpy as np

from regression import FISTA
from problems import linmodel
from signal_approximator import signal_approximator, signal_approximator_sparse




class problem(object):

    """
    A problem class with a smooth component, and a seminorm component stored in self.semi
    """

    def __init__(self, data, semi):
        self.semi = semi
        self.initialize(data)

    @property
    def output(self):
        r = self.Y - np.dot(self.X, self.coefs)
        return self.coefs.copy(), r
    
    def set_coefs(self, coefs):
        self._coefs = coefs

    def get_coefs(self):
        return self._coefs
    coefs = property(get_coefs, set_coefs)

    def set_response(self,Y):
        self._Y = Y

    def get_response(self):
        return self._Y
    Y = property(get_response, set_response)

    @property
    def default_coefs(self):
        return np.zeros(self.p)

    def obj(self, beta):
        return self.obj_smooth(beta) + self.obj_rough(beta)

    def obj_rough(self, beta):
        return self.semi.evaluate(beta)

    def proximal(self, coefs, grad, L):
        return self.semi.proximal(coefs, grad, L)

class squaredloss(problem):

    """
    A class for combining squared error loss with a general seminorm
    """


    def initialize(self, data):
        """
        Generate initial tuple of arguments for update.
        """

        if len(data) == 2:
            self.X = data[0]
            self.Y = data[1]
            self.n, self.p = self.X.shape
            self.semi.p = self.p
        else:
            raise ValueError("Data tuple not as expected")

        if hasattr(self,'initial_coefs'):
            self.set_coefs(self.initial_coefs)
        else:
            self.set_coefs(self.default_coefs)
            

    def obj_smooth(self, beta):
        #Smooth part of objective
        return ((self.Y - np.dot(self.X, beta))**2).sum() / 2. 

    def grad(self, beta):
        XtXbeta = np.dot(self.X.T, np.dot(self.X, beta)) 
        return XtXbeta - np.dot(self.Y,self.X) 

class seminorm(object):
    """
    A seminorm container class for storing/combining seminorm_atom classes
    """
    def __init__(self, atoms):
        if isinstance(atoms,list):
            self.atoms = atoms
        else:
            self.atoms = [atoms]
        self.M = len(self.atoms)
        self.p = 1

    def __add__(self,y):
        #Combine two seminorms
        atoms = self.atoms + y.atoms
        return seminorm(atoms)

    @property
    def segments(self):
        if not hasattr(self, '_segments'):
            idx = 0
            self._segments = []
            for i in range(self.M):
                self._segments.append(slice(idx, idx+self.p))
                idx += self.p
        return self._segments                

    def evaluate(self, beta):
        out = 0.
        for i in range(self.M):
            out += self.atoms[i].evaluate(beta)
        return out
    
    def proximal(self, z, g, L):
        #Solve the proximal function for the seminorms
        v = z - g / L
        out = np.zeros(v.shape)

        for i in range(self.M):
            out += self.atoms[i].solve_prox(v, L)
        return out
        """
        for i, segment in enumerate(self.segments):

            #segment = self.segments[i]
            #print "seg", segment, v[segment].shape            
            #v[segment] =

        return v
        """
    
class seminorm_atom(object):

    """
    A seminorm atom class
    """
    def __init__(self, D, l=1.):
        self.D = D
        self.l = l
        
class l1norm(seminorm_atom):

    """
    The l1 norm
    """

    def __init__(self, l=1.):
        self.l = l
        
    def evaluate(self, beta):
        return self.l * np.fabs(beta).sum()

    def solve_prox(self, v,  L):
        return np.sign(v) * np.maximum(np.fabs(v)-self.l/L, 0)

class genl1norm(seminorm_atom):

    """
    The generalized l1 norm \|D\beta\|_1
    """

    dualcontrol = {'max_its':5000,
                   'tol':1.0e-8,
                   'restart':np.inf}

    def __init__(self, D, l=1.):
        self.D = D
        self.l = l
        self.dual = signal_approximator_sparse((self.D, np.zeros(self.D.shape[1])))
        self.dualopt = FISTA(self.dual)

    def evaluate(self, beta):
        return self.l * np.fabs(self.D.matvec(beta)).sum()

    def solve_prox(self, v, L):
        self.dual.set_response(v)
        self.dual.assign_penalty(l1=self.l/L)
        self.dualopt.debug = True
        self.dualopt.fit(**self.dualcontrol)
        return self.dualopt.output[0]




"""


class james_stein(proximal):

    def solve(self, v, L):
        normv = np.linalg.norm(v)
        if normv <= l:
            return v
        else:
            return v * (l / normv)


class truncate(proximal):
    
    #Vector truncated to have norm <= l (projection onto
    #Euclidean ball of radius l.
    
    normV = norm2(V)
    if normV <= l:
        return V
    else:
        return V * (l / normV)
    
    def james_stein(V, l):
        
        #James-Stein estimator:
        
        V - truncate(V, l)
        normV = norm2(V)
        return max(1 - l / normV, 0) * V

"""
    
"""
class group_approximator(signal_approximator):


    @property
    def default_penalties(self):
        return {}

    def initialize(self, data):

        #Generate initial tuple of arguments for update.

        if len(data) == 2:
            penalties = {}
            self.Ds = []
            self.segments = []
            idx = 0
            for i, v in enumerate(data[0]):
                D, penalty = v
                D = np.atleast_2d(D)
                self.Ds.append(D) 
                self.segments.append(slice(idx, idx+D.shape[0]))
                idx += D.shape[0]
                penalties['V%d' % i] =penalty
            self.assign_penalty(**penalties)
            self.Y = data[1]
            self.D = np.vstack(self.Ds)
            self.n = self.Y.shape[0]
        else:
            raise ValueError("Data tuple not as expected")

        if hasattr(self,'initial_coefs'):
            self.set_coefs(self.initial_coefs)
        else:
            self.set_coefs(self.default_coefs)

    @property
    def default_coefs(self):
        return np.zeros(self.D.shape[0])

    def compute_penalty(self, beta):
        pen = 0
        for i, D in enumerate(self.Ds):
            pen += self.penalties['V%d' % i] * norm2(np.dot(D, beta))
        return pen

    def obj(self, dual):
        beta = self.Y - np.dot(dual, self.D)
        return ((self.Y - beta)**2).sum() / 2. + self.compute_penalty(beta)

    def grad(self, dual):
        dual = np.asarray(dual)
        return np.dot(self.D, np.dot(dual, self.D) - self.Y)

    def proximal(self, z, g, L):
        v = z - g / L
        for i, segment in enumerate(self.segments):
            l = self.penalties['V%d' % i]
            v[segment] = truncate(v[segment], l/L)
        return v

    def f(self, dual):
        #Smooth part of objective
        beta = self.Y - np.dot(dual, self.D)
        return ((self.Y - beta)**2).sum() / 2.
                            

    @property
    def output(self):
        r = np.dot(self.coefs, self.D) 
        return self.Y - r, r

class group_lasso(linmodel):

    dualcontrol = {'max_its':50,
                   'tol':1.0e-06}

    def initialize(self, data):

        #Generate initial tuple of arguments for update.
        
        if len(data) == 3:
            self.X = data[0]
            self.Dv = data[1]
            penalties = {}
            for i, v in enumerate(data[1]):
                _, penalty = v
                penalties['V%d' % i] =penalty
            self.assign_penalty(**penalties)
            self.Y = data[2]
            self.n, self.p = self.X.shape
        else:
            #raise ValueError("Data tuple not as expected")

        self.dual = group_approximator((self.Dv, self.Y))
        self.dualopt = FISTA(self.dual)

        self.m = self.dual.D.shape[0]

        if hasattr(self,'initial_coefs'):
            self.set_coefs(self.initial_coefs)
        else:
            self.set_coefs(self.default_coefs)

    @property
    def default_penalties(self):

        #Default penalty for Lasso: a single
        #parameter problem.

        #XXX maybe use a recarray for the penalties
        return {}

    @property
    def default_coefs(self):
        return np.zeros(self.p)

    # this is the core generalized LASSO functionality

    def obj(self, beta):
        return ((self.Y - np.dot(self.X, beta))**2).sum() / 2. + self.dual.compute_penalty(beta)

    def grad(self, beta):
        return np.dot(self.X.T, np.dot(self.X, beta) - self.Y)

    def proximal(self, z, g, L):
        v = z - g / L
        self.dual.set_response(v)
        #XXX this is painful -- maybe do it with a recarray multiplication?
        penalties = {}
        for i in range(len(self.dual.Ds)):
            penalties['V%d' % i] = self.penalties['V%d' % i] / L
        self.dual.assign_penalty(**penalties)
        self.dualopt.fit(**self.dualcontrol)
        return self.dualopt.output[0]

    @property
    def output(self):
        r = self.Y - np.dot(self.X, self.coefs) 
        return self.coefs, r


def norm2(V):

    #The Euclidean norm of a vector.

    return np.sqrt((V**2).sum())

def truncate(V, l):

    #Vector truncated to have norm <= l (projection onto
    #Euclidean ball of radius l.

    normV = norm2(V)
    if normV <= l:
        return V
    else:
        return V * (l / normV)

def james_stein(V, l):

    #James-Stein estimator:

    V - truncate(V, l)
    normV = norm2(V)
    return max(1 - l / normV, 0) * V

"""
# The API is to have a gengrad class in each module.
# In this module, this is the signal_approximator

#gengrad = group_lasso
