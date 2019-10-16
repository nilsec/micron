import subprocess
import os

script_name = "script.py"

scripts = [os.path.join(os.path.abspath("."), f) for f in os.listdir(".")]
scripts = [os.path.join(d, script_name) for d in scripts if os.path.isdir(d) and "setup" in d]

command = ""
for f in scripts:
    command += "python {} & ".format(f)

command = command[:-2]

subprocess.run(command, shell=True)
