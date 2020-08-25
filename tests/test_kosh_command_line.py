from __future__ import print_function
import os
from koshbase import KoshTest
import shutil
import shlex
from subprocess import Popen, PIPE
import numpy


def run_cmd(cmd):
    print(cmd)
    cmd = shlex.split(cmd)
    p = Popen(cmd, stdout=PIPE, stderr=PIPE)
    o, e = p.communicate()
    assert(p.returncode == 0)
    return o.decode(
        "utf-8").strip().split("\n"), e.decode("utf-8").strip().split("\n")


class KoshTestDataset(KoshTest):
    def test_kosh_command(self):
        shutil.copy("tests/baselines/sina/data.sqlite", "cmd_line.sql")
        store, kosh_db = self.connect(
            db_uri="cmd_line.sql", dataset_record_type="obs")
        # Search the all store
        o, e = run_cmd("kosh search -s 'cmd_line.sql' -d obs")
        self.assertEqual(len(o), 10)
        o, e = run_cmd("kosh search -s 'cmd_line.sql' -d obs PARAM1=143.557")
        self.assertEqual(len(o), 1)
        o, e = run_cmd("kosh search -s 'cmd_line.sql' -d obs PARAM1>241.289")
        self.assertEqual(len(o), 3)
        o, e = run_cmd("kosh search -s 'cmd_line.sql' -d obs PARAM1>=241.289")
        self.assertEqual(len(o), 4)
        o, e = run_cmd("kosh search -s 'cmd_line.sql' -d obs PARAM1<241.289")
        self.assertEqual(len(o), 6)
        o, e = run_cmd("kosh search -s 'cmd_line.sql' -d obs PARAM1<=241.289")
        self.assertEqual(len(o), 7)
        o, e = run_cmd(
            "kosh search -s 'cmd_line.sql' -d obs PARAM1=DataRange(140,445)")
        self.assertEqual(len(o), 8)
        o, e = run_cmd(
            "kosh add -s 'cmd_line.sql' -d obs -i '2020-03-11-13-45-23' PARAM1=156 PARAM2=.2 PARAM3=something")
        ds = store.open('2020-03-11-13-45-23')
        self.assertEqual(ds.PARAM1, 156)
        self.assertEqual(ds.PARAM2, .2)
        self.assertEqual(ds.PARAM3, "something")
        o, e = run_cmd(
            "kosh remove -s 'cmd_line.sql' -d obs -i '2020-03-11-13-45-23' -f")
        with self.assertRaises(Exception):
            ds = store.open('2020-03-11-13-45-23')
        o, e = run_cmd(
            "kosh associate -s 'cmd_line.sql' -d obs -i '2019-04-05-02-11-29'"
            " -u tests/baselines/node_extracts2/node_extracts2.hdf5 -m hdf5")
        ds = store.open('2019-04-05-02-11-29')
        self.assertEqual(len(ds._associated_data_), 1)
        o, e = run_cmd(
            "kosh dissociate -s 'cmd_line.sql' -d obs -i '2019-04-05-02-11-29'"
            " -u tests/baselines/node_extracts2/node_extracts2.hdf5")
        self.assertEqual(len(ds._associated_data_), 0)
        o, e = run_cmd(
            "kosh associate -s 'cmd_line.sql' -d obs -i '2019-04-05-02-11-29'"
            " -u tests/baselines/node_extracts2/node_extracts2.hdf5 -m hdf5")
        o, e = run_cmd(
            "kosh features -s 'cmd_line.sql' -d obs -i '2019-04-05-02-11-29'")
        self.assertTrue("node/metrics_4" in o[1])
        npyfile = "test_kosh_cmd.npy"
        if os.path.exists(npyfile):
            os.remove(npyfile)
        o, e = run_cmd(
            "kosh extract -s 'cmd_line.sql' -d obs -i '2019-04-05-02-11-29'"
            " -f node/metrics_4 zone/metrics_4 --dump {npyfile}".format(npyfile=npyfile))
        self.assertTrue(os.path.exists(npyfile))
        n4, z4 = numpy.load(npyfile)
        self.assertEqual(n4.shape, (18,))
        self.assertEqual(z4.shape, (18,))
