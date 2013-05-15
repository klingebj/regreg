"""
RegReg: A package to solve regularized regression problems
"""

import os, sys
import string

from Cython.Compiler import Main

def cython_extension(srcfile):
    options = Main.CompilationOptions(include_path=[os.path.join(os.path.abspath(os.path.dirname(__file__)), 'include')])
    Main.compile(srcfile, options=options)

def configuration(parent_package='',top_path=None):
    from numpy.distutils.misc_util import Configuration
    config = Configuration('regreg',parent_package,top_path)
#     config.add_extension('atoms.projl1_cython',
#                          sources = ["atoms/projl1_cython.c"],
#                          )
#     config.add_extension('atoms.mixed_lasso_cython',
#                          sources = ["atoms/mixed_lasso_cython.c"],
#                          )
#     config.add_extension('atoms.piecewise_linear',
#                          sources = ["atoms/piecewise_linear.c"],
#                          )
    config.add_subpackage('atoms')
    config.add_subpackage('affine')
    config.add_subpackage('problems')
    config.add_subpackage('smooth')

    return config

if __name__ == '__main__':


    
    from numpy.distutils.core import setup

    c = configuration(top_path='',
                      ).todict()
    setup(**c)
