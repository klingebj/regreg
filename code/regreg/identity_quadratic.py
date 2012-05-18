"""
This module contains a single class that is meant to represent
a quadratic of the form

.. math::

   \frac{\kappa}{2} \|x-\gamma\|^2_2 + \alpha^Tx + c

with :math:`\kappa, \gamma, \alpha, c` = (coef, offset, linear_term, constant_term).
"""

from numpy.linalg import norm

class identity_quadratic(object):

    def __init__(self, coef, offset, linear_term, constant_term=0):
        if coef is None:
            self.coef = 0
        else:
            self.coef = coef
        self.offset = offset
        self.linear_term = linear_term
        if constant_term is None:
            self.constant_term = 0
        else:
            self.constant_term = constant_term
        if self.coef is not None or self.linear_term is not None:
            self.anything_to_return = True
        else:
            self.anything_to_return = False

    def objective(self, x, mode='both'):
        coef, offset, linear_term = self.coef, self.offset, self.linear_term
        cons = self.constant_term
        if linear_term is None:
            linear_term = 0
        if offset is not None:
            r = x - offset
        else:
            r = x
        if mode == 'both':
            if linear_term is not None:
                return (norm(r)**2 * coef / 2. + (linear_term * x).sum() 
                        + cons, coef * r + linear_term)
            else:
                return (norm(r)**2 * coef / 2. + cons,
                        coef * r)
        elif mode == 'func':
            if linear_term is not None:
                return norm(r)**2 * coef / 2. + (linear_term * x).sum() + cons
            else:
                return norm(r)**2 * coef / 2. + cons
        elif mode == 'grad':
            if linear_term is not None:
                return coef * r + linear_term
            else:
                return coef * r
        else:
            raise ValueError("Mode incorrectly specified")
                        
    def __repr__(self):
        return 'identity_quadratic(%f, %s, %s, %f)' % (self.coef, str(self.offset), str(self.linear_term), self.constant_term)

    def __add__(self, other):
        """
        Return an identity quadratic given by the sum in the obvious way. it has offset of 0,
        would be nice to have None, but there are some 
        places we are still multiplying by -1
        """
        if not (other is None or isinstance(other, identity_quadratic)):
            raise ValueError('can only add None or other identity_quadratic')


        if other is None:
            return self
        else:
            sc = self.collapsed()
            oc = other.collapsed()
            newq = identity_quadratic(sc.coef + oc.coef, 0, 
                                      sc.linear_term + oc.linear_term,
                                      sc.constant_term + oc.constant_term)
            return newq 

    def __getitem__(self, slice):
        '''
        Return a new quadratic restricted to the variables in slice
        with constant_term=0.
        '''
        return identity_quadratic(self.coef,
                                  self.offset[slice],
                                  self.linear_term[slice],
                                  0)

    def collapsed(self):
        """
        Return an identity quadratic with offset of 0,
        would be nice to have None, but there are some 
        places we are still multiplying by -1
        """

        if self.coef is None:
            coef = 0
        else:
            coef = self.coef

        linear_term = 0
        constant_term = self.constant_term
        if constant_term is None: 
            constant_term = 0 
        if self.offset is not None:
            linear_term -= coef * self.offset
            constant_term += coef * norm(self.offset)**2/2.
        if self.linear_term is not None:
            linear_term += self.linear_term

        return identity_quadratic(coef, 0, linear_term, constant_term)
