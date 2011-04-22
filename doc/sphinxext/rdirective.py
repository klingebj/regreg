import sys, os, cStringIO
try:
    from hashlib import md5
except ImportError:
    from md5 import md5

from docutils.parsers.rst import directives
from rrunner import shell

def rplot_directive(name, arguments, options, content, lineno,
                   content_offset, block_text, state, state_machine):


    if len(arguments) == 1: # it is a filename
        content = [file(arguments[0]).read()]

    m = md5()
    m.update('\n'.join(content))
    hash = m.hexdigest()[-10:]

    if not os.path.exists(os.path.join("images", "inline")):
        os.makedirs(os.path.join("images", "inline"))
    shell.process('png("%s", width=600, height=600)' % os.path.join("images", "inline", "%s.png" % hash))
    shell.astext() # flush
    
    for line in content:
        shell.process(line)
    output = shell.astext().split('\n')

    shell.process('dev.off()')
    shell.astext() # flush

    if options.has_key('silent'):
        silent = True
    else:
        silent = False
        
    if options.has_key('source'):
        lines = content
    else:
        lines = output

    if len(lines) and not silent:
        r_lines = ['.. code-block:: r', '']
        r_lines.extend([u'    %s'% line for line in lines])
        r_lines.extend(['.. image:: images/inline/%s.png' % hash, ''])

        state_machine.insert_input(
            r_lines, state_machine.input_lines.source(0))

    return []

def rcode_directive(name, arguments, options, content, lineno,
                   content_offset, block_text, state, state_machine):

    for line in content:
        shell.process(line)

    output = shell.astext().split('\n')

    if options.has_key('silent'):
        silent = True
    else:
        silent = False

    if options.has_key('source'):
        lines = content
    else:
        lines = output

    if len(lines) and not silent:
        r_lines = ['.. code-block:: r', '']
        r_lines.extend([u'    %s'%line for line in lines])
        r_lines.append('')

        state_machine.insert_input(
            r_lines, state_machine.input_lines.source(0))

    return []

options = {'silent':directives.flag,
           'source':directives.flag}

def setup(app):
    setup.app = app
    app.add_directive('rplot', rplot_directive, True, (0, 2, 0), **options)
    app.add_directive('rcode', rcode_directive, True, (0, 2, 0), **options)


