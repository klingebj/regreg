"""
A setup script for RegReg
"""

from numpy.distutils.misc_util import Configuration

def configuration(parent_package='',top_path=None, package_name='regreg'):
    
    config = Configuration('regreg',parent_package,top_path)

    
    config.add_extension('regreg',
                         sources = ["regreg.c"],
                         )

    
    return config

if __name__ == '__main__':
    
    from numpy.distutils.core import setup
    
    setup(**configuration(top_path='',
                          package_name=package_name,
                          ).todict())
    

                        
