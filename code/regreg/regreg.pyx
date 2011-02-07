# cython: profile=True

import numpy as np
cimport numpy as np
import time

## Local imports
import nesterov_smooth
import l1smooth

## Compile-time datatypes
DTYPE_float = np.float
ctypedef np.float_t DTYPE_float_t
DTYPE_int = np.int
ctypedef np.int_t DTYPE_int_t


## Include other cython source files
include "subfunctions.pxi"
include "problems.pxi"

## Include algorithm code
include "cwpath.pxi"
include "direct.pxi"
include "ista.pxi"
include "fista.pxi"
include "nesterov_eps.pxi"
#include "nesterov_smooth.py"

def regreg(data, problemtype, algorithm, **kwargs):
    #Create optimization algorithm
    #try:
    return algorithm(data, problemtype, **kwargs)
    #except:
    #    raise ValueError("Error creating algorithm class")


