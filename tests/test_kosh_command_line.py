from __future__ import print_function
import os
from koshbase import KoshTest
import shutil
import shlex
from subprocess import Popen, PIPE
import numpy


def run_cmd(cmd, verbose=False):
    if verbose:
        print(cmd)
    cmd = shlex.split(cmd)
    p = Popen(cmd, stdout=PIPE, stderr=PIPE)
    o, e = p.communicate()
    if p.returncode != 0:
        print("OOOOPSY:", o.decode())
        print("OOOOPSY:", e.decode())
    if verbose:
        print("OUT:", o)
        print("ERR:", e)
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
        os.remove(kosh_db)
        os.remove(npyfile)

    def test_dissociate_dead_files(self):
        store, kosh_db = self.connect()
        ds = store.create()
        ds.associate("setup.py", "py")  # real one
        ds.associate("blablabla.py", "py")  # dead one
        ds.associate("blablbla.hdf5", "hdf5")  # dead one
        # real one
        ds.associate(
            "tests/baselines/node_extracts2/node_extracts2.hdf5", "hdf5")
        ds.associate("README.md", "md")  # real
        ds.associate("REEEEDME.mmmmdddd", "md")  # fake one
        verbose = False
        self.assertEqual(len(ds.search()), 6)
        # first test cleanup python files only
        self.assertEqual(len(ds.search(mime_type="py")), 2)
        # Dry run first
        cmd = "kosh cleanup_files -s '{}' -d blah --dry-run mime_type=py".format(
            kosh_db)
        o, e = run_cmd(cmd, verbose=verbose)
        # Let's make sure it's still all here
        self.assertEqual(len(ds.search()), 6)
        self.assertEqual(len(ds.search(mime_type="py")), 2)
        cmd = "kosh cleanup_files -s '{}' -d blah mime_type=py".format(kosh_db)
        o, e = run_cmd(cmd, verbose=verbose)
        # Let's make sure only one py file was removed
        self.assertEqual(len(ds.search()), 5)
        self.assertEqual(len(ds.search(mime_type="py")), 1)
        # Let's clean it all
        cmd = "kosh cleanup_files -s '{}' -d blah ".format(kosh_db)
        o, e = run_cmd(cmd, verbose=verbose)
        self.assertEqual(len(ds.search()), 3)
        self.assertEqual(len(ds.search(mime_type="py")), 1)
        self.assertEqual(len(ds.search(mime_type="md")), 1)
        self.assertEqual(len(ds.search(mime_type="hdf5")), 1)
        os.remove(kosh_db)


if __name__ == "__main__":
    A = KoshTestDataset()
    for nm in dir(A):
        if nm[:4] == "test":
            fn = getattr(A, nm)
            print(nm, fn)
            fn()
