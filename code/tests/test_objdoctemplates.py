""" Test objective doc templater
"""

from regreg.objdoctemplates import objective_doc_templater

from regreg.doctemplates import (
    doc_templater,
    doc_template_provider,
    doc_template_user)

from nose.tools import assert_equal, assert_raises


def test_obj_doc_templater():
    @objective_doc_templater(dict(b='two'))
    class C(object):
        objective_template = r'''f(%(var)s)'''
        objective_vars = {'var': 'p'}

        @doc_template_user
        @doc_template_provider
        def fa(self, arg):
            """A docstring %(objective)s"""

        @doc_template_user
        @doc_template_provider
        def fb(self, arg):
            """B docstring %(b)s"""

    # Providing a template sets the docstring to None
    assert_equal(C().fa.__doc__, 'A docstring f(p)')
    assert_equal(C().fb.__doc__, "B docstring two")
