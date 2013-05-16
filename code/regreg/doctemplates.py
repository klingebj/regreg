""" Decorators for making and using docstring templates on methods

The doctest code uses class decorators and therefore needs at least Python 2.6.

If you have Python 2.5 you can decorate the class without using class
decorators::

    class C(object):
        @doc_template_provider
        def f(self):
            "A docstring %(var)s"
    C = doc_templater()(C)


Examples
--------

We can set a docstring template from a method docstring

>>> @doc_templater()
... class C(object):
...     @doc_template_provider
...     def f(self):
...         "A docstring %(var)s"

This wipes out the docstring and stores it for future use:

>>> C().f.__doc__ is None
True

We can use it with the ``doc_template_user`` decorator on the method

>>> @doc_templater(dict(var='something'))
... class D(C):
...     @doc_template_user
...     def f(self):
...         pass

>>> D().f.__doc__
'A docstring something'

The ``doc_dict`` parameter to the class decorator sets parameters for the
docstring for the class.  These override parameters from inherited classes:

>>> @doc_templater(dict(var='something else'))
... class E(D):
...     @doc_template_user
...     def f(self):
...         pass

>>> E().f.__doc__
'A docstring something else'

>>> @doc_templater(dict(var='something else again'))
... class F(E):
...     @doc_template_user
...     def f(self):
...         pass

>>> F().f.__doc__
'A docstring something else again'

The same method docstring can be a template and template user:

>>> @doc_templater(dict(foo='something else entirely'))
... class G(object):
...     @doc_template_user
...     @doc_template_provider
...     def f2(self):
...         "Look at %(foo)s"

>>> G().f2.__doc__
'Look at something else entirely'
"""

def doc_template_user(m):
    """ Decorator to mark method `m` as using docstring template

    If ``klass`` is the enclosing class of `m`, and ``klass`` is decorated with
    ``doc_templater``, then the docstring for `m` will be givem by
    ``klass._doc_templates[m.func_name] % klass._doc_dict``
    """
    m._uses_doc_template = True
    return m


def doc_template_provider(m):
    """ Decorator to mark docstring from method `m` as being a doc template

    We label the docstring of `m` (``m.func_doc``) to be put into the
    ``_doc_templates`` attribute of the enclosing class, when the klass is
    decorated with the ``doc_templater`` decorator.
    """
    m._doc_template = m.func_doc
    m.func_doc = None
    return m


def doc_templater(doc_dict=None, doc_error=True):
    """ Return class decorator to record and provide doc templates

    Parameters
    ----------
    doc_dict : dict
        Extra key / value pairs for this class to fill template docstrings.
        Parameters in this input dict override parameters from inherited
        classes.
    doc_error : {True, False}, optional
        Whether to raise error in case of non-empty docstring that will be
        thrown away by using docstring template, where the docstring has not
        itself been marked for use as a template.

    Returns
    -------
    kdec : function
        Class decorator
    """
    if doc_dict is None:
        doc_dict = {}
    def kdec(klass):
        if hasattr(klass, '_doc_templates'): # inherited
            klass._doc_templates = klass._doc_templates.copy()
        else:
            klass._doc_templates = {}
        if hasattr(klass, '_doc_dict'): # inherited
            klass_doc_dict = klass._doc_dict.copy()
            klass_doc_dict.update(doc_dict)
        else:
            klass_doc_dict = doc_dict
        for obj in klass.__dict__.values():
            if hasattr(obj, '_doc_template') and hasattr(obj, 'func_name'):
                klass._doc_templates[obj.func_name] = obj._doc_template
            if hasattr(obj, '_uses_doc_template') and hasattr(obj, 'func_name'):
                if doc_error and not obj.func_doc is None:
                    raise ValueError("Refusing to discard unexpected docstring for %s" % repr(obj) +
                                     " - set `doc_error` to False if you want " + 
                                     "to allow this")
                template = klass._doc_templates[obj.func_name]
                obj.func_doc = template % klass_doc_dict
        klass._doc_dict = klass_doc_dict
        return klass
    return kdec
