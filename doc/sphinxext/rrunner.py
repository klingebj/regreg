import sys, cStringIO, os
from tempfile import mkstemp

from IPython.irunner import InteractiveRunner, PythonRunner

class RRunner(InteractiveRunner):
    """Interactive R runner that has an R_PROFILE
    that specifies an optional prompt and continue
    which is useful to distinguish the prompt from
    R's use of the ">" to mean "greater than" in, i.e.
    "summary.lm".

    The profile can be added to by supplying a
    string or file-like object, to which the call to "options" is appended.
    """

    def __init__(self, program = "R" ,args=None,out=sys.stdout,echo=True,
                 profile='', prompt=">>>R", continue_prompt=">>>+"):
        """New runner."""
        prompt_profile = '\noptions(prompt="%s ", continue="%s ")\n' % (prompt, continue_prompt)

        if type(profile) == type(''):
            profilestr = profile + prompt_profile
        else:
            profilestr = profile.read() + prompt_profile
            
        prompts = [r'%s' % prompt, r'%s' % continue_prompt]
        self.profile = file(mkstemp(suffix='Rprofile')[1], mode='w')
        self.profile.write(profilestr)
        self.profile.close()
        os.environ['R_PROFILE'] = self.profile.name

        InteractiveRunner.__init__(self,program,prompts,args,out,echo)

        # Run a one-liner to "flush" things a little
        # so the tmp file can be removed.
        # It seems to hang otherwise -- perhaps
        # because the options change the prompts?

        self.echo = False
        self.run_source('flushing...\n')
        self.echo = True
        os.remove(self.profile.name)
        
class EmbeddedRShell:
    def __init__(self):

        self.cout = cStringIO.StringIO()
        self.R = RRunner(out=self.cout)

    def process(self, line):

        self.R.run_source('%s\n'% line)

    def astext(self):
        self.cout.seek(0)
        s = self.cout.read()
        # R uses fancy quotes in their summaries
        # and so the strings should be interpreted
        # as unicode
        s = s.decode('utf-8') 
        s = s.replace(self.R.prompts[0], ">")
        s = s.replace(self.R.prompts[1], "+")
        self.cout.truncate(0)
        return s

shell = EmbeddedRShell()
