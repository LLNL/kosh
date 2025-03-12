from __future__ import print_function
import os
import shlex
from subprocess import Popen, PIPE
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))  # noqa
from koshbase import KoshTest  # noqa


class KoshTestStore(KoshTest):

    # Code path
    pth = os.path.dirname(__file__)
    pth = os.path.dirname(os.path.dirname(pth))
    # Tests path
    test_pth = os.path.join(pth, "tests")

    def send_command(self, cmd, shell=False):

        print(f"Command: {cmd}")
        self.maxDiff = None

        if not sys.platform.startswith("win"):
            cmd = shlex.split(cmd)
        P = Popen(cmd,
                  stdout=PIPE,
                  stderr=PIPE,
                  shell=shell)
        out, e = P.communicate()
        out = out.decode("utf-8")
        e = e.decode("utf-8")
        print(out, e)
        # self.assertIn("Total Time to complete:", out, "Did not complete")

    def test_kosh_scaling_maestro(self):
        # Ensemble Test

        # Script path
        yaml_path = os.path.join(self.test_pth, "parallel", "kosh_scaling_maestro.yaml")
        pgen_path = os.path.join(self.test_pth, "parallel", "kosh_scaling_pgen.py")

        # maestro run tests/parallel/kosh_scaling_maestro.yaml --pgen tests/parallel/kosh_scaling_pgen.py -y
        # With --lock-strategy RFileLock
        #     2 parallel runs with --ensembles 2 --datasets 2
        #         Total Time to complete: 0:01:02.995092
        #         Total Time to complete: 0:00:50.245494
        #     3 parallel runs with --ensembles 3 --datasets 3
        #         Total Time to complete: 0:04:49.546512
        #         Total Time to complete: 0:03:43.376198
        #         Total Time to complete: 0:04:49.750664
        #     4 parallel runs with --ensembles 4 --datasets 4
        #         Total Time to complete: 0:15:47.718486
        #         Total Time to complete: 0:16:43.870328
        #         Total Time to complete: 0:14:27.504426
        #         Total Time to complete: 0:15:15.394995
        #     5 parallel runs with --ensembles 5 --datasets 5
        #         Total Time to complete: 0:44:01.426139
        #         Total Time to complete: 0:41:36.120883
        #         Total Time to complete: 0:39:55.211173
        #         Total Time to complete: 0:38:37.345506
        #         Total Time to complete: 0:44:56.977541
        # With --lock-strategy RetryOnly
        #     2 parallel runs with --ensembles 2 --datasets 2
        #         Total Time to complete: 0:00:30.429637
        #         Total Time to complete: 0:00:31.284906
        #     3 parallel runs with --ensembles 3 --datasets 3
        #         Total Time to complete: 0:01:33.871474
        #         Total Time to complete: 0:01:35.490541
        #         Total Time to complete: 0:01:33.882631
        #     4 parallel runs with --ensembles 4 --datasets 4
        #         Total Time to complete: 0:03:55.400093
        #         Total Time to complete: 0:03:50.518424
        #         Total Time to complete: 0:03:56.670136
        #         Total Time to complete: 0:03:54.860387
        #     5 parallel runs with --ensembles 5 --datasets 5
        #         Total Time to complete: 0:08:04.702222
        #         Total Time to complete: 0:08:04.789991
        #         Total Time to complete: 0:08:04.591291
        #         Total Time to complete: 0:08:15.845611
        #         Total Time to complete: 0:08:27.977732
        #     10 parallel runs with --ensembles 10 --datasets 10
        #         Total Time to complete: 1:31:35.300161
        #         Total Time to complete: 1:31:33.409552
        #         Total Time to complete: 1:31:39.584701
        #         Total Time to complete: 1:31:09.328600
        #         Total Time to complete: 1:30:59.356357
        #         Total Time to complete: 1:31:22.978819
        #         Total Time to complete: 1:31:26.143434
        #         Total Time to complete: 1:35:13.974000
        #         Total Time to complete: 1:32:26.120776
        #         Total Time to complete: 1:32:46.681288
        cmd = f"maestro run {yaml_path} " + \
              f"--pgen {pgen_path} -y"

        self.send_command(cmd)

    def kosh_scaling_merlin(self):
        # Ensemble Test

        # Script path
        yaml_path = os.path.join(self.test_pth, "kosh_scaling_merlin.yaml")
        pgen_path = os.path.join(self.test_pth, "kosh_scaling_pgen.py")

        # merlin run tests/kosh_scaling_merlin.yaml --pgen tests/kosh_scaling_pgen.py
        # merlin run-workers tests/kosh_scaling.yaml
        cmd = f"merlin run {yaml_path} " + \
              f"--pgen {pgen_path} " + \
              "&& " + \
              f"merlin run-workers {yaml_path}"

        self.send_command(cmd, shell=True)
