from __future__ import print_function
import unittest
import os
import shlex
from subprocess import Popen, PIPE
import sys


class TestFlake8(unittest.TestCase):

    def testFlake8(self):
        # Code path
        self.maxDiff = None
        pth = os.path.dirname(__file__)
        pth = os.path.dirname(pth)
        code_pth = os.path.join(pth, "kosh")
        # Tests path
        test_pth = os.path.join(pth, "tests")
        print()
        print()
        print()
        print()
        print("---------------------------------------------------")
        print("RUNNING: flake8 on directory %s" % pth)
        print("---------------------------------------------------")
        print()
        print()
        print()
        print()
        cmd = "flake8 --show-source --statistics " +\
              "--exclude scripts/* " +\
              "--max-line-length=120 {} scripts {} ".format(code_pth, test_pth) +\
              "--per-file-ignores='kosh/kosh_command.py:E265'"
        if not sys.platform.startswith("win"):
            cmd = shlex.split(cmd)
        P = Popen(cmd,
                  stdout=PIPE,
                  stderr=PIPE)
        out, e = P.communicate()
        out = out.decode("utf-8")
        print(out, e)
        if out != "":
            print(out)
        self.assertEqual(out, "")
