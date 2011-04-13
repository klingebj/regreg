import sys, os, shutil, imp, warnings, cStringIO, re
try:
    from hashlib import md5
except ImportError:
    from md5 import md5

from docutils.parsers.rst import directives
try:
    # docutils 0.4
    from docutils.parsers.rst.directives.images import align
except ImportError:
    # docutils 0.5
    from docutils.parsers.rst.directives.images import Image
    align = Image.align
import sphinx


sphinx_version = sphinx.__version__.split(".")
# The split is necessary for sphinx beta versions where the string is
# '6b1'
sphinx_version = tuple([int(re.split('[a-z]', x)[0])
                        for x in sphinx_version[:2]])


import IPython
from IPython.Shell import MatplotlibShell


class EmbeddedSphinxShell:
    def __init__(self):

        self.cout = cStringIO.StringIO()

        IPython.Shell.Term.cout = self.cout
        IPython.Shell.Term.cerr = self.cout
        argv = []
        self.user_ns = {}
        self.user_glocal_ns = {}

        self.IP = IPython.ipmaker.make_IPython(
            argv, self.user_ns, self.user_glocal_ns, embedded=True,
            #shell_class=IPython.Shell.InteractiveShell,
            shell_class=MatplotlibShell,
            rc_override=dict(colors = 'NoColor'))

    def process(self, line):

        self.cout.write('%s%s\n'%(self.IP.outputcache.prompt1, line))
        stdout = sys.stdout
        sys.stdout = self.cout
        self.IP.push(line)
        sys.stdout = stdout


    def astext(self):
        self.cout.seek(0)
        s = self.cout.read()
        self.cout.truncate(0)
        return s



shell = EmbeddedSphinxShell()


def ipython_directive(name, arguments, options, content, lineno,
                   content_offset, block_text, state, state_machine):

    for line in content:
        shell.process(line)

    lines = shell.astext().split('\n')
    if len(lines):
        ipy_lines = ['.. sourcecode:: ipython', '']
        ipy_lines.extend(['    %s'%line for line in lines])
        ipy_lines.append('')

        state_machine.insert_input(
            ipy_lines, state_machine.input_lines.source(0))

    return []

def setup(app):
    setup.app = app
    options = {}
    app.add_directive('ipython', ipython_directive, True, (0, 2, 0), **options)


