.. _agenda:

RegReg development
~~~~~~~~~~~~~~~~~~

We currently host and manage our code at github. You can find our repo at http://github.com/klingebj/regreg.

Near-term development goals include:

* Adding tutorials and implementations for problems in `Elements of Statistical Learning <http://www-stat.stanford.edu/~tibs/ElemStatLearn>`_

* Adding a Python implementation of a generalized `interior point algorithm <http://stanford.edu/~boyd/papers/l1_trend_filter.html>`_ for banded problems 

  * For use with block-wise descent

  * Can generalize to atoms beyond l1norm

* Adding `ADMM <http://www.stanford.edu/~boyd/papers/pdf/admm_notes_draft.pdf>`_ support

  * This is similar the block-wise descent framework

  * Use `ipython's muliengine interface <http://ipython.scipy.org/doc/manual/html/parallel/parallel_multiengine.html>`_ for parallel implementation

* Writing a guide to writing your own smooth function atoms

* Writing a guide to writing your own seminorm atoms

* Adding a cross-validation framework

Long-term goals include:

* Extending current framework for matrix problems

* Continually improving constructor syntax to make framework as natural as possible