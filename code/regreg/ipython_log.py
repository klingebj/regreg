#log# Automatic Logger file. *** THIS MUST BE THE FIRST LINE ***
#log# DO NOT CHANGE THIS LINE OR THE TWO BELOW
#log# opts = Struct({'__allownew': True, 'logfile': 'ipython_log.py'})
#log# args = []
#log# It is safe to make manual edits below here.
#log#-----------------------------------------------------------------------
_ip.magic("logstart ")

import numpy as np
A = np.random.standard_normal((100,50))
b = np.random.standard_normal(100)
import regreg.api as rr
l2 = rr.l2norm.linear(A, offset=-b, lagrange=1.)
pos = rr.constrained_positive_part(50, lagrange=lam)
lam = 2
pos = rr.constrained_positive_part(50, lagrange=lam)
c = rr.container(l2, pos)
solver = rr.FISTA(c)
solver.fit(debug=True)
_ip.magic("hist -n")
