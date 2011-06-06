.. _download:

Downloading and installing RegReg
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The RegReg source code is hosted at 

http://github.com/klingebj/regreg

RegReg depends on the following Python tools

* `NumPy <http://numpy.scipy.org>`_

* `SciPy <http://www.scipy.org>`_

You can clone the RegReg github repo using::


     git clone git://github.com/klingebj/regreg.git

Then installation is a simple call to python::

     cd regreg
     python setup.py install --prefix=MYDIR

where MYDIR is a site-packages directory you can write to. This directory will need to be on your PYTHONPATH for you to import RegReg. That's it!

Testing your installation
-------------------------

There is a small but growing suite of tests that be easily checked using `nose <http://somethingaboutorange.com/mrl/projects/nose/1.0.0/>`_::

     cd regreg/code/tests
     nosetests

