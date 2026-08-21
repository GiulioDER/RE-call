"""Stands in for a command that hangs, and spawns a child that outlives it."""

import subprocess
import sys
import time

# The grandchild inherits this process's stdout pipe, which is the whole difficulty:
# killing the direct child does not close it.
subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
time.sleep(30)
