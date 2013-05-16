""" Implement doc templates using objective templates and vars
"""

from .doctemplates import doc_templater


def objective_doc_templater(doc_dict=None, doc_errors=True):
    """ Return doc_templater class generator for klasses with objective docs

    Check for specific case where class being decorated has attributes
    ``object_template`` and ``objective_vars``.  Set new entry ``objective`` into class
    ``doc_dict`` using these class attributes.  Then continue decorating as for
    ``doct_templater``.
    """
    if doc_dict is None:
        doc_dict = {}
    def obj_kdec(klass):
        if (hasattr(klass, 'objective_template') and
            hasattr(klass, 'objective_vars')):
            doc_dict['objective'] = (klass.objective_template %
                                     klass.objective_vars)
            for k, v in klass.objective_vars.items():
                doc_dict[k] = v
        return doc_templater(doc_dict, doc_errors)(klass)
    return obj_kdec

