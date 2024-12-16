from __future__ import print_function
import os
from koshbase import KoshTest
import shlex
from subprocess import Popen, PIPE
import sys
import uuid


class KoshTestStore(KoshTest):

    # Code path
    pth = os.path.dirname(__file__)
    pth = os.path.dirname(pth)
    # Tests path
    test_pth = os.path.join(pth, "tests")

    def send_command(self, cmd, shell=False, verbose=False):

        if verbose:
            print("Received command:", cmd)

        self.maxDiff = None

        if not sys.platform.startswith("win"):
            cmd = shlex.split(cmd)

        P = Popen(cmd,
                  stdout=PIPE,
                  stderr=PIPE,
                  shell=shell)
        out, e = P.communicate()
        out = out.decode("utf-8")
        print(out, e)
        self.assertIn("Total Time to complete:", out, "Did not complete")

    def test_kosh_scaling_local(self):
        # Make sure everything works before we scale it up

        lock_path = os.path.join(os.path.expanduser("~"), ".kosh.test.lock")
        if os.path.exists(lock_path):  # in case sopmething went weird before and the lock file is still here
            os.remove(lock_path)
        # Script path
        verbose = True
        log_level = 20
        script_path = os.path.join(self.test_pth, "kosh_scaling.py")

        # Without lock-strategy single run
        # rm -rf run_0*
        # python tests/kosh_scaling.py -s run_0.sql --run-number 0 --datasets 5 --ensembles 5 --lock-strategy None
        # --ensembles 1 --datasets 1 Total Time to complete: 0:00:04.297681
        # --ensembles 2 --datasets 2 Total Time to complete: 0:00:24.493327
        # --ensembles 3 --datasets 3 Total Time to complete: 0:01:55.359536
        # --ensembles 4 --datasets 4 Total Time to complete: 0:03:11.737136
        # --ensembles 5 --datasets 5 Total Time to complete: 0:08:20.810381
        # --ensembles 10 --datasets 10 Total Time to complete: 1:21:41.972174
        cmd = f"python {script_path} " + \
              f"--store 'run_{uuid.uuid1().hex}.sql' " + \
              "--run-number 0 " + \
              "--ensembles 3 " + \
              "--datasets 3 " + \
              "--retries 2 " + \
              "--lock-strategy None " + \
              f"--log-level {log_level}"

        self.send_command(cmd, verbose=verbose)

        # With lock-strategy single run
        # rm -rf run_0*
        # python tests/kosh_scaling.py -s run_0.sql --run-number 0 --datasets 5 --ensembles 5 --lock-strategy RFileLock
        # --ensembles 1 --datasets 1 Total Time to complete: 0:00:04.577028
        # --ensembles 2 --datasets 2 Total Time to complete: 0:00:26.316382
        # --ensembles 3 --datasets 3 Total Time to complete: 0:02:16.385435
        # --ensembles 4 --datasets 4 Total Time to complete: 0:03:22.155168
        # --ensembles 5 --datasets 5 Total Time to complete: 0:07:42.208587
        # --ensembles 10 --datasets 10 Total Time to complete: 1:35:30.319923

        if os.path.exists(lock_path):  # in case sopmething went weird before and the lock file is still here
            os.remove(lock_path)
        cmd = f"python {script_path} " + \
              f"--store 'run_{uuid.uuid1().hex}.sql' " + \
              "--run-number 0 " + \
              "--ensembles 3 " + \
              "--datasets 3 " + \
              "--retries 2 " + \
              "--lock-strategy RFileLock " + \
              f"--lock-path {lock_path} " + \
              f"--log-level {log_level}"

        self.send_command(cmd, verbose=verbose)
