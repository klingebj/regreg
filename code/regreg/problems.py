import numpy as np

class linmodel(object):

    @property
    def output(self):
        return self.coefs, self.r

    def __init__(self, data, penalties={}, initial_coefs=None):
        
        self.assign_penalty(**penalties)
        if initial_coefs is not None: 
            self.initial_coefs = initial_coefs
        self.initialize(data)
    
    def initialize(self, data):

        """
        Generate initial tuple of arguments for update.
        """

        if len(data) == 2:
            self.X = data[0]
            self.n, self.p = self.X.shape
            self.Y = data[1]
        else:
            raise ValueError("Data tuple not as expected")

        if hasattr(self,'initial_coefs'):
            self.set_coefs(self.initial_coefs)
        else:
            self.set_coefs(self.default_coefs)

    @property
    def default_coefs(self):
        return np.zeros(self.p)

    def assign_penalty(self, **params):
        """
        Abstract method for assigning penalty parameters.
        """
        if not hasattr(self, "penalties") and hasattr(self, "default_penalties"):
            self.penalties = self.default_penalties

        for key in params:
            self.penalties[key] = params[key]
        
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


