"""
This module contains a single class that is meant to represent
a quadratic of the form

.. math::

   \frac{\kappa}{2} \|x-\mu\|^2_2 + \beta^Tx + \gamma

with :math:`\kappa, \mu, \beta, \gamma` = (coef, center, linear_term, constant_term).
"""

from copy import copy

from numpy.linalg import norm
from numpy import all


class identity_quadratic(object):

    def __init__(self, coef, center, linear_term, constant_term=0):
        if coef is None:
            self.coef = 0
        else:
            self.coef = coef
        self.center = center
        self.linear_term = linear_term
        if constant_term is None:
            self.constant_term = 0
        else:
            self.constant_term = constant_term

    @property
    def iszero(self):
        return all([self.coef in [0, None],
                    self.center is None or all(self.center == 0),
                    self.linear_term is None or all(self.linear_term == 0),
                    self.constant_term in [0, None]])

    def __copy__(self):
        return identity_quadratic(self.coef,
                                  copy(self.center),
                                  copy(self.linear_term),
                                  copy(self.constant_term))

    def noneify(self):
        '''
        replace zeros with nones
        '''
        if self.coef is None:
            self.coef = 0
        if self.constant_term is None:
            self.constant_term = 0
        if self.linear_term is not None and all(self.linear_term == 0):
            self.linear_term = None
        if self.center is not None and all(self.center_term == 0):
            self.center_term = None

    def zeroify(self):
        for a in ['coef', 'center', 'linear_term', 'constant_term']:
            if getattr(self, a) is None:
                setattr(self, a, 0)

    def recenter(self, offset):

        if offset is not None and all(offset == 0):
            offset = None

        if offset is not None:
            cpq = copy(self)
            cpq.center += offset
            cpq = cpq.collapsed()
            return offset, cpq
        else:
            return None, self.collapsed()


    def objective(self, x, mode='both'):
        coef, center, linear_term = self.coef, self.center, self.linear_term
        cons = self.constant_term
        if linear_term is None:
            linear_term = 0
        if center is not None:
            r = x - center
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
        return 'identity_quadratic(%f, %s, %s, %f)' % (self.coef, str(self.center), str(self.linear_term), self.constant_term)

    def __add__(self, other):
        """
        Return an identity quadratic given by the sum in the obvious way. it has center of 0,
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
                                  self.center[slice],
                                  self.linear_term[slice],
                                  0)

    def collapsed(self):
        """
        Return an identity quadratic with center of 0,
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
        if self.center is not None:
            linear_term -= coef * self.center
            constant_term += coef * norm(self.center)**2/2.
        if self.linear_term is not None:
            linear_term += self.linear_term

        return identity_quadratic(coef, 0, linear_term, constant_term)

    def latexify(self, var='x', idx=''):
        self.zeroify()
        v = ' '
        if self.coef != 0:
            v += r'\frac{\kappa_{%s}}{2} ' % idx
            if not all(self.center == 0):
                v += r'\|%s-\mu_{%s}\|^2_2 + ' % (var, idx)
        if not all(self.linear_term == 0):
            v += r' \beta_{%s}^T%s ' % (idx, var)
        if self.constant_term != 0:
            v += r' + \gamma_{%s} ' % idx
        return v

