from __future__ import print_function
import unittest
import os
import shlex
from subprocess import Popen, PIPE


class TestFlake8(unittest.TestCase):

    def testFlake8(self):
        # Code path
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
              "--exclude scripts/kosh_commands.py, scripts/kosh, scripts/init_sina.py " +\
              "--max-line-length=120 {} scripts {} ".format(code_pth, test_pth)
        P = Popen(shlex.split(cmd),
                  stdout=PIPE,
                  stderr=PIPE)
        out, e = P.communicate()
        out = out.decode("utf-8")
        if out != "":
            print(out)
        self.assertEqual(out, "")
