"""
RegReg: A package to solve regularized regression problems
"""

import os, sys
import string

from Cython.Compiler import Main

def configuration(parent_package='',top_path=None):
    from numpy.distutils.misc_util import Configuration
    config = Configuration('atoms',parent_package,top_path)
    config.add_extension('projl1_cython',
                         sources = ["projl1_cython.c"],
                         )
    config.add_extension('mixed_lasso_cython',
                         sources = ["mixed_lasso_cython.c"],
                         )
    config.add_extension('piecewise_linear',
                         sources = ["piecewise_linear.c"],
                         )
    return config

if __name__ == '__main__':


    
    from numpy.distutils.core import setup

    c = configuration(top_path='',
                      ).todict()
    setup(**c)
