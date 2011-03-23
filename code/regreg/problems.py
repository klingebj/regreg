import numpy as np

class squaredloss(object):

    """
    A class for combining squared error loss with a general seminorm
    """

    def __init__(self, X, Y):
        """
        Generate initial tuple of arguments for update.
        """
        self.X = X
        self.Y = Y
        self.n, self.p = self.X.shape

    def obj_smooth(self, beta):
        #Smooth part of objective
        return ((self.Y - np.dot(self.X, beta))**2).sum() / 2. 

    def grad(self, beta):
        XtXbeta = np.dot(self.X.T, np.dot(self.X, beta)) 
        return XtXbeta - np.dot(self.Y,self.X) 

    def add_seminorm(self, seminorm, initial=None, smooth_multiplier=1):
        return seminorm.problem(self.obj_smooth, self.grad, smooth_multiplier=smooth_multiplier,
                                initial=initial)
